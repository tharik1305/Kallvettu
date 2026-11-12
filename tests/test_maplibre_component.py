"""
Unit & Integration Tests for MapLibre GL JS + OpenFreeMap Geographic Atlas Component.
"""

import os
import unittest
from services.geo_atlas.models import GeoInscriptionRecord, LocationPrecision
from services.geo_atlas.database import AtlasDatabase
from services.geo_atlas.map_builder import (
    MAPLIBRE_STYLES,
    records_to_markers_geojson,
    build_districts_geojson,
    build_states_geojson,
    compute_maplibre_view_state,
)
from services.geo_atlas.maplibre_component import OPENFREEMAP_STYLES


class TestMapLibreAtlasComponent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = AtlasDatabase()

    def test_openfreemap_style_endpoints_no_proprietary_keys(self):
        """Verify OpenFreeMap vector tile styles are strictly open-source without Google/Mapbox keys."""
        expected = ["liberty", "bright", "positron", "dark"]
        for key in expected:
            self.assertIn(key, OPENFREEMAP_STYLES)
            url = OPENFREEMAP_STYLES[key]
            self.assertTrue(url.startswith("https://tiles.openfreemap.org/styles/"))
            # Must NOT use proprietary Google or Mapbox endpoints
            self.assertNotIn("googleapis.com", url)
            self.assertNotIn("mapbox.com", url)
            self.assertNotIn("key=", url)
            self.assertNotIn("token=", url)

    def test_component_html_bundle_exists_and_contains_maplibre(self):
        """Verify maplibre_component/index.html is intact and contains MapLibre and OpenFreeMap."""
        comp_dir = os.path.join(
            os.path.dirname(__file__), "..", "services", "geo_atlas", "maplibre_component"
        )
        html_path = os.path.join(comp_dir, "index.html")
        self.assertTrue(os.path.isfile(html_path))

        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("maplibre-gl.js", content)
        self.assertIn("maplibre-gl.css", content)
        self.assertIn("tiles.openfreemap.org/styles/liberty", content)
        self.assertIn("streamlit:setComponentValue", content)
        self.assertIn("streamlit:componentReady", content)
        self.assertIn("clusters", content)
        self.assertIn("OpenStreetMap", content)

    def test_records_to_markers_geojson_anti_fabrication(self):
        """Verify records_to_markers_geojson strictly excludes district_only & unknown records."""
        all_recs = self.db.query_inscriptions()
        self.assertGreater(len(all_recs), 0)

        # 5 out of 7 default records have mappable coordinates (2 are district_only)
        fc = records_to_markers_geojson(all_recs, selected_id="MT_TNJ_001")
        self.assertEqual(fc["type"], "FeatureCollection")

        features = fc["features"]
        self.assertEqual(len(features), 5)

        for feat in features:
            self.assertEqual(feat["type"], "Feature")
            self.assertEqual(feat["geometry"]["type"], "Point")
            coords = feat["geometry"]["coordinates"]
            self.assertEqual(len(coords), 2)
            # Longitude, Latitude ordering for GeoJSON
            lon, lat = coords[0], coords[1]
            self.assertTrue(68.0 <= lon <= 98.0)
            self.assertTrue(6.0 <= lat <= 38.0)

            props = feat["properties"]
            self.assertIn(props["precision"], [LocationPrecision.VERIFIED.value, LocationPrecision.APPROXIMATE.value])
            # Must NOT be district-only
            self.assertNotEqual(props["precision"], LocationPrecision.DISTRICT_ONLY.value)

            if props["inscription_id"] == "MT_TNJ_001":
                self.assertTrue(props["is_selected"])
            else:
                self.assertFalse(props["is_selected"])

    def test_build_districts_geojson(self):
        """Verify district boundary FeatureCollection generation with inscriptional counts."""
        dist_counts = {
            "Thanjavur": 1,
            "Madurai": 2,
        }
        fc = build_districts_geojson("Tamil Nadu", dist_counts, selected_district="Thanjavur")
        self.assertEqual(fc["type"], "FeatureCollection")
        self.assertGreater(len(fc["features"]), 30)

        # Check properties of Thanjavur
        thanjavur_feat = next(
            (f for f in fc["features"] if f["properties"].get("district") == "Thanjavur"),
            None,
        )
        self.assertIsNotNone(thanjavur_feat)
        self.assertEqual(thanjavur_feat["properties"]["count"], 1)
        self.assertTrue(thanjavur_feat["properties"]["is_selected"])

    def test_build_states_geojson(self):
        """Verify state boundary FeatureCollection with 36 States/UTs and counts."""
        state_counts = {"Tamil Nadu": 5, "Karnataka": 2}
        fc = build_states_geojson(state_counts, selected_state="Tamil Nadu")
        self.assertEqual(fc["type"], "FeatureCollection")
        self.assertEqual(len(fc["features"]), 36)

        tn_feat = next(
            (f for f in fc["features"] if f["properties"].get("state") == "Tamil Nadu"),
            None,
        )
        self.assertIsNotNone(tn_feat)
        self.assertEqual(tn_feat["properties"]["count"], 5)
        self.assertTrue(tn_feat["properties"]["is_selected"])

    def test_compute_maplibre_view_state(self):
        """Verify MapLibre [longitude, latitude] view state computation across hierarchies."""
        # 1. Default India
        v_india = compute_maplibre_view_state()
        self.assertEqual(v_india["center"], [79.2, 21.8])
        self.assertAlmostEqual(v_india["zoom"], 4.15)

        # 2. State view
        v_state = compute_maplibre_view_state(selected_state="Tamil Nadu")
        self.assertAlmostEqual(v_state["zoom"], 6.2)
        self.assertTrue(76.0 <= v_state["center"][0] <= 80.5)
        self.assertTrue(8.0 <= v_state["center"][1] <= 14.0)

        # 3. District view
        v_dist = compute_maplibre_view_state(selected_state="Tamil Nadu", selected_district="Madurai")
        self.assertAlmostEqual(v_dist["zoom"], 8.2)

        # 4. Inscription site view
        rec = self.db.get_inscription("MT_TNJ_001")
        self.assertIsNotNone(rec)
        v_site = compute_maplibre_view_state(selected_record=rec)
        self.assertEqual(v_site["center"], [float(rec.longitude), float(rec.latitude)])
        self.assertAlmostEqual(v_site["zoom"], 10.2)

        # 5. Point picker view
        v_pick = compute_maplibre_view_state(picker_coords=(9.9252, 78.1198))
        self.assertEqual(v_pick["center"], [78.1198, 9.9252])
        self.assertAlmostEqual(v_pick["zoom"], 8.8)

    def test_maplibre_event_payload_simulation(self):
        """Verify MapLibre event payload structure for Streamlit event loop integration."""
        from services.geo_atlas.geo_loader import normalize_state_name, normalize_district_name

        # 1. Inscription click event
        marker_ev = {
            "event_type": "marker_click",
            "inscription_id": "MT_TNJ_001",
            "title_en": "Brihadisvara Temple Inscription",
            "state": "Tamil Nadu",
            "district": "Thanjavur",
            "coordinates": [10.7828, 79.1318],
            "timestamp": 123456789,
        }
        self.assertEqual(marker_ev["event_type"], "marker_click")
        self.assertEqual(marker_ev["inscription_id"], "MT_TNJ_001")
        self.assertEqual(normalize_state_name(marker_ev["state"]), "Tamil Nadu")
        self.assertEqual(normalize_district_name(marker_ev["state"], marker_ev["district"]), "Thanjavur")

        # 2. District selection event
        dist_ev = {
            "event_type": "district_select",
            "district": "Madurai",
            "state": "Tamil Nadu",
            "coordinates": [9.9252, 78.1198],
            "timestamp": 123456790,
        }
        self.assertEqual(dist_ev["event_type"], "district_select")
        self.assertEqual(dist_ev["district"], "Madurai")

        # 3. State selection event
        state_ev = {
            "event_type": "state_select",
            "state": "Karnataka",
            "coordinates": [15.3173, 75.7139],
            "timestamp": 123456791,
        }
        self.assertEqual(state_ev["event_type"], "state_select")
        self.assertEqual(normalize_state_name(state_ev["state"]), "Karnataka")

        # 4. Coordinate pick event
        pick_ev = {
            "event_type": "coordinate_pick",
            "coordinates": [10.0190, 78.2536],
            "timestamp": 123456792,
        }
        self.assertEqual(pick_ev["event_type"], "coordinate_pick")
        self.assertEqual(len(pick_ev["coordinates"]), 2)


if __name__ == "__main__":
    unittest.main()
