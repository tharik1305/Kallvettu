"""
Automated Unit & Integration Tests for Kallvettu's India Epigraphical Geographic Atlas.

Covers:
1. Administrative GeoJSON dataset integrity, source metadata, & lazy state district loading.
2. Coordinate validation (WGS84 range, NaN/Inf, null-island rejection, India bounds, point-in-district check).
3. Anti-fabrication enforcement: district_only & unknown records NEVER receive or plot coordinates.
4. Separated 4-pillar geographic provenance (discovery, collection, current repository, photo capture).
5. SQLite schema creation, CRUD operations, audit trail history, and curated preset seeding.
6. Multi-facet search & filtering, district/state counts (including unpinned records), and PyDeck clustering/selection parsing.
"""

import os
import tempfile
import unittest
from typing import Dict, Any

from services.geo_atlas import (
    LocationPrecision,
    LocationContext,
    GeoInscriptionRecord,
    AtlasFilterCriteria,
    load_geo_metadata,
    load_all_districts_geojson,
    load_india_states_geojson,
    load_state_districts_geojson,
    get_all_state_names,
    get_districts_for_state,
    find_admin_unit_for_point,
    normalize_state_name,
    normalize_district_name,
    validate_coordinates,
    validate_record_provenance,
    AtlasDatabase,
    cluster_pin_records,
    build_atlas_deck,
    extract_pydeck_selection,
)


class TestGeoDatasetAndValidation(unittest.TestCase):
    def test_india_states_and_lazy_district_loading(self):
        meta = load_geo_metadata()
        self.assertGreaterEqual(meta.get("total_states_uts", 0), 36)
        self.assertIn("source_repository", meta)

        states_gj = load_india_states_geojson()
        self.assertEqual(states_gj.get("type"), "FeatureCollection")
        self.assertGreaterEqual(len(states_gj.get("features", [])), 36)

        all_states = get_all_state_names()
        self.assertIn("Tamil Nadu", all_states)
        self.assertIn("Ladakh", all_states)
        self.assertIn("Karnataka", all_states)

        # Lazy per-state district loading
        tn_gj = load_state_districts_geojson("Tamil Nadu")
        self.assertGreaterEqual(len(tn_gj.get("features", [])), 30)
        tn_districts = get_districts_for_state("Tamil Nadu")
        self.assertIn("Thanjavur", tn_districts)
        self.assertIn("Madurai", tn_districts)
        self.assertIn("Kancheepuram", tn_districts)

    def test_load_all_districts_geojson_integrity_and_fallback(self):
        # 1. Primary composite load
        all_districts = load_all_districts_geojson()
        self.assertEqual(all_districts.get("type"), "FeatureCollection")
        features = all_districts.get("features", [])
        self.assertGreaterEqual(len(features), 700)

        # Inspect feature structure & properties
        first = features[0]
        self.assertEqual(first.get("type"), "Feature")
        self.assertIn("geometry", first)
        self.assertIn("properties", first)
        props = first["properties"]
        self.assertTrue("district" in props or "dtname" in props)
        self.assertTrue("st_nm" in props or "state_name" in props)

        # 2. Test fallback aggregation when composite file path does not exist
        import services.geo_atlas.geo_loader as loader
        original_comp_path = loader.ALL_DISTRICTS_GEOJSON_PATH
        try:
            loader.ALL_DISTRICTS_GEOJSON_PATH = os.path.join(tempfile.gettempdir(), "nonexistent_comp.geojson")
            loader.load_all_districts_geojson.cache_clear()
            aggregated = loader.load_all_districts_geojson()
            self.assertEqual(aggregated.get("type"), "FeatureCollection")
            agg_feats = aggregated.get("features", [])
            self.assertGreaterEqual(len(agg_feats), 700)
            # Verify properties preserved from per-state files
            sample = agg_feats[0]
            self.assertIn("properties", sample)
            self.assertTrue("district" in sample["properties"] or "dtname" in sample["properties"])
        finally:
            loader.ALL_DISTRICTS_GEOJSON_PATH = original_comp_path
            loader.load_all_districts_geojson.cache_clear()

        # 3. Test safe empty fallback when both composite and state directory are missing
        original_state_dir = loader.PER_STATE_GEO_DIR
        try:
            loader.ALL_DISTRICTS_GEOJSON_PATH = os.path.join(tempfile.gettempdir(), "nonexistent_comp.geojson")
            loader.PER_STATE_GEO_DIR = os.path.join(tempfile.gettempdir(), "nonexistent_states_dir")
            loader.load_all_districts_geojson.cache_clear()
            empty_res = loader.load_all_districts_geojson()
            self.assertEqual(empty_res, {"type": "FeatureCollection", "features": []})
        finally:
            loader.ALL_DISTRICTS_GEOJSON_PATH = original_comp_path
            loader.PER_STATE_GEO_DIR = original_state_dir
            loader.load_all_districts_geojson.cache_clear()

    def test_historical_district_and_state_aliases(self):
        self.assertEqual(normalize_state_name("Orissa"), "Odisha")
        self.assertEqual(normalize_district_name("Tamil Nadu", "Kanchipuram"), "Kancheepuram")
        self.assertEqual(normalize_district_name("Tamil Nadu", "Sivagangai"), "Sivaganga")

    def test_coordinate_validation_bounds_and_point_in_district(self):
        # Invalid WGS84 bounds
        bad_lat = validate_coordinates(95.0, 79.0)
        self.assertFalse(bad_lat.is_valid)
        self.assertFalse(bad_lat.within_wgs84_bounds)

        # Null-island sentinel rejected
        zero = validate_coordinates(0.0, 0.0)
        self.assertFalse(zero.is_valid)

        # Out-of-India valid WGS84 raises warning
        outside = validate_coordinates(51.5074, -0.1278, check_boundary=True)
        self.assertTrue(outside.is_valid)
        self.assertFalse(outside.within_india_bounds)
        self.assertTrue(len(outside.warnings) > 0)

        # Valid Brihadisvara Temple Thanjavur inside Thanjavur district
        tnj = validate_coordinates(10.7828, 79.1318, state="Tamil Nadu", district="Thanjavur")
        self.assertTrue(tnj.is_valid)
        self.assertTrue(tnj.within_india_bounds)
        self.assertTrue(tnj.point_in_state)
        self.assertTrue(tnj.point_in_district)
        self.assertEqual(tnj.matched_district, "Thanjavur")

        # Mismatched district warning (Thanjavur point checked against Madurai)
        mismatch = validate_coordinates(10.7828, 79.1318, state="Tamil Nadu", district="Madurai")
        self.assertTrue(mismatch.is_valid)
        self.assertFalse(mismatch.point_in_district)
        self.assertTrue(any("District boundary check" in w for w in mismatch.warnings))

    def test_point_admin_lookup(self):
        unit = find_admin_unit_for_point(10.01904, 78.25367, state_hint="Tamil Nadu")
        self.assertIsNotNone(unit)
        self.assertEqual(unit["state"], "Tamil Nadu")
        self.assertEqual(unit["district"], "Madurai")

    def test_anti_fabrication_rule_for_district_only_and_unknown(self):
        # Attempting to store coordinates on district_only raises validation failure unless stripped
        fake_centroid_rec = GeoInscriptionRecord(
            inscription_id="TEST_NO_CENTROID",
            title_en="District Only Inscription",
            state="Tamil Nadu",
            district="Tiruvannamalai",
            latitude=12.2253,
            longitude=79.0747,
            location_precision=LocationPrecision.DISTRICT_ONLY.value,
        )
        ok, errors, _ = validate_record_provenance(fake_centroid_rec, strip_disallowed_coords=False)
        self.assertFalse(ok)
        self.assertTrue(any("must NOT have exact marker coordinates" in e for e in errors))

        # Stripping removes fabricated coordinates cleanly
        ok_stripped, _, warnings = validate_record_provenance(fake_centroid_rec, strip_disallowed_coords=True)
        self.assertTrue(ok_stripped)
        self.assertIsNone(fake_centroid_rec.latitude)
        self.assertIsNone(fake_centroid_rec.longitude)
        self.assertFalse(fake_centroid_rec.has_mappable_coordinates())
        self.assertTrue(len(warnings) > 0)


class TestAtlasDatabaseAndMapBuilder(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "test_atlas.db")
        self.db = AtlasDatabase(db_path=self.db_path, auto_seed=True)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_curated_seeding_integrity_and_no_fabricated_coords(self):
        recs = self.db.query_inscriptions()
        self.assertEqual(len(recs), 7)

        # Chengam Nadukal must stay district_only with None coords
        nadukal = self.db.get_inscription("VT_CGM_001")
        self.assertIsNotNone(nadukal)
        self.assertEqual(nadukal.location_precision, LocationPrecision.DISTRICT_ONLY.value)
        self.assertEqual(nadukal.district, "Tiruvannamalai")
        self.assertIsNone(nadukal.latitude)
        self.assertIsNone(nadukal.longitude)
        self.assertFalse(nadukal.has_mappable_coordinates())

        # Regional Arulmolidevan stays unknown with None coords
        arl = self.db.get_inscription("MC_ARL_001")
        self.assertIsNotNone(arl)
        self.assertEqual(arl.location_precision, LocationPrecision.UNKNOWN.value)
        self.assertIsNone(arl.latitude)
        self.assertIsNone(arl.longitude)

        # Thanjavur Brihadisvara is verified and pinned
        tnj = self.db.get_inscription("MT_TNJ_001")
        self.assertIsNotNone(tnj)
        self.assertTrue(tnj.has_mappable_coordinates())
        self.assertEqual(tnj.district, "Thanjavur")
        self.assertEqual(tnj.district_code, "620")

    def test_separated_location_contexts_persistence_and_crud(self):
        custom = GeoInscriptionRecord(
            inscription_id="KA_MSK_001",
            title_en="Maski Minor Rock Edict Reference",
            title_ta="மாஸ்கி கல்வெட்டு",
            script="Brahmi",
            period="c. 3rd Century BCE",
            dynasty="Maurya",
            material="Granite Boulder",
            country="India",
            state="Karnataka",
            district="Raichur",
            historical_district_name="Hyderabad State / Raichur",
            sub_district_taluk="Lingsugur",
            village="Maski",
            archaeological_site="Maski Rock Outcrop",
            latitude=15.9620,
            longitude=76.6570,
            location_precision=LocationPrecision.APPROXIMATE.value,
            discovery_source="C. Beadon (1915)",
            collection_source="Archaeological Survey of Hyderabad",
            current_repository="In-situ Protected ASI Monument, Maski",
            publication_reference="Hultzsch CII Vol. I",
            discovery_location=LocationContext(
                place_name="Maski Goldfields Boulder",
                state="Karnataka",
                district="Raichur",
                latitude=15.9620,
                longitude=76.6570,
            ),
            collection_location=LocationContext(
                place_name="Hyderabad Archaeological Survey Archive",
                state="Telangana",
                district="Hyderabad",
            ),
            current_repository_location=LocationContext(
                place_name="Maski Protected Site",
                state="Karnataka",
                district="Raichur",
            ),
            photograph_capture_location=LocationContext(
                place_name="Digital Field Squeeze Unit",
                state="Karnataka",
                district="Raichur",
            ),
        )
        res = self.db.upsert_inscription(custom, validate_spatial=True)
        self.assertEqual(res["action"], "INSERT")

        loaded = self.db.get_inscription("KA_MSK_001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.discovery_location.district, "Raichur")
        self.assertEqual(loaded.collection_location.district, "Hyderabad")
        self.assertEqual(loaded.current_repository_location.place_name, "Maski Protected Site")
        self.assertEqual(loaded.photograph_capture_location.place_name, "Digital Field Squeeze Unit")

        # Audit history recorded
        audit = self.db.get_audit_history("KA_MSK_001")
        self.assertGreaterEqual(len(audit), 1)
        self.assertEqual(audit[0]["action"], "INSERT")

        # Multi-facet query filter
        ka_recs = self.db.query_inscriptions(AtlasFilterCriteria(state="Karnataka"))
        self.assertEqual(len(ka_recs), 1)
        self.assertEqual(ka_recs[0].inscription_id, "KA_MSK_001")

        # Delete
        self.assertTrue(self.db.delete_inscription("KA_MSK_001"))
        self.assertIsNone(self.db.get_inscription("KA_MSK_001"))

    def test_district_counts_include_district_only_without_map_pins(self):
        dist_counts = self.db.get_district_counts()
        # Tiruvannamalai has Chengam Nadukal (district_only = 1, pinned = 0, total = 1)
        tvm = dist_counts.get(("Tamil Nadu", "Tiruvannamalai"))
        self.assertIsNotNone(tvm)
        self.assertEqual(tvm["total"], 1)
        self.assertEqual(tvm["pinned"], 0)
        self.assertEqual(tvm["district_only"], 1)

        all_recs = self.db.query_inscriptions()
        singles, clusters = cluster_pin_records(all_recs, cell_deg=0.0)
        pin_ids = {p["inscription_id"] for p in singles}
        self.assertEqual(len(clusters), 0)
        self.assertNotIn("VT_CGM_001", pin_ids)
        self.assertNotIn("MC_ARL_001", pin_ids)
        self.assertIn("MT_TNJ_001", pin_ids)

    def test_clustering_and_pydeck_deck_and_selection_parsing(self):
        recs = self.db.query_inscriptions()
        state_counts = self.db.get_state_counts()
        dist_counts = self.db.get_district_counts()

        # Verify clustering merges close sites (e.g. Mangulam & Pulankurichi at large cell_deg=1.5)
        singles, clusters = cluster_pin_records(recs, cell_deg=1.5)
        total_mapped = len(singles) + sum(c["cluster_count"] for c in clusters)
        self.assertEqual(total_mapped, 5)
        self.assertGreaterEqual(len(clusters), 1)

        # Build PyDeck Deck for India view & State-drilled view
        deck_india = build_atlas_deck(recs, state_counts, dist_counts)
        layer_ids_india = [l.id for l in deck_india.layers]
        self.assertIn("india_states", layer_ids_india)
        self.assertNotIn("state_districts", layer_ids_india)

        deck_tn = build_atlas_deck(
            recs,
            state_counts,
            dist_counts,
            selected_state="Tamil Nadu",
            selected_district="Thanjavur",
        )
        layer_ids_tn = [l.id for l in deck_tn.layers]
        self.assertIn("india_states", layer_ids_tn)
        self.assertIn("state_districts", layer_ids_tn)
        self.assertIn("site_markers", layer_ids_tn)

        # Selection parsing priority checks
        fake_state_click: Dict[str, Any] = {
            "selection": {"objects": {"india_states": [{"properties": {"st_nm": "Tamil Nadu"}}]}}
        }
        parsed_state = extract_pydeck_selection(fake_state_click)
        self.assertEqual(parsed_state["kind"], "state")
        self.assertEqual(parsed_state["state"], "Tamil Nadu")

        fake_dist_click: Dict[str, Any] = {
            "selection": {
                "objects": {
                    "state_districts": [{"properties": {"st_nm": "Tamil Nadu", "district": "Madurai"}}],
                    "india_states": [{"properties": {"st_nm": "Tamil Nadu"}}],
                }
            }
        }
        parsed_dist = extract_pydeck_selection(fake_dist_click)
        self.assertEqual(parsed_dist["kind"], "district")
        self.assertEqual(parsed_dist["district"], "Madurai")

        fake_marker_click: Dict[str, Any] = {
            "selection": {
                "objects": {
                    "site_markers": [{"inscription_id": "MT_TNJ_001", "state": "Tamil Nadu", "district": "Thanjavur"}],
                    "state_districts": [{"properties": {"st_nm": "Tamil Nadu", "district": "Thanjavur"}}],
                }
            }
        }
        parsed_marker = extract_pydeck_selection(fake_marker_click)
        self.assertEqual(parsed_marker["kind"], "inscription")
        self.assertEqual(parsed_marker["inscription_id"], "MT_TNJ_001")


if __name__ == "__main__":
    unittest.main()

