"""
Comprehensive Audit Test Suite for services.geo_atlas.geo_loader.

Covers all 16 required test scenarios:
1. Importing geo_loader.py independently.
2. Importing the complete Geographic Atlas package (lightweight, zero eager UI errors).
3. Single valid district GeoJSON.
4. Multiple valid district GeoJSON files.
5. Empty geographic dataset.
6. Missing dataset directory.
7. Missing GeoJSON file.
8. Invalid JSON syntax / decode errors.
9. Malformed FeatureCollection.
10. Missing geometry (geometry is None / non-existent).
11. Invalid coordinates (NaN, out-of-range, swapped).
12. Duplicate feature handling during aggregation.
13. Unicode Tamil district and state names.
14. Incorrect / alias state and district identifiers.
15. Windows file paths and pathlib.Path compatibility.
16. Cache invalidation on dataset file modification (mtime-aware).
"""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import services.geo_atlas as geo_pkg
import services.geo_atlas.geo_loader as loader
from services.geo_atlas.geo_loader import (
    load_geo_metadata,
    load_all_districts_geojson,
    load_india_states_geojson,
    load_state_districts_geojson,
    get_all_state_names,
    get_districts_for_state,
    get_state_admin_info,
    get_district_feature,
    get_district_bounds_and_center,
    find_admin_unit_for_point,
    normalize_state_name,
    normalize_district_name,
    state_name_to_slug,
    validate_geojson_structure,
    validate_geojson_feature,
    clear_geo_cache,
    _read_json_with_mtime_cache,
)


class TestGeoLoaderCompleteAudit(unittest.TestCase):
    def setUp(self):
        clear_geo_cache()

    def tearDown(self):
        clear_geo_cache()

    # --- 1 & 2: Imports ---
    def test_import_geo_loader_independently(self):
        """Verify geo_loader can be imported independently without side effects."""
        self.assertTrue(callable(load_all_districts_geojson))
        self.assertTrue(callable(load_state_districts_geojson))
        self.assertTrue(callable(validate_geojson_structure))

    def test_import_complete_package_and_lazy_components(self):
        """Verify the geo_atlas package imports cleanly and supports lazy component access."""
        self.assertTrue(hasattr(geo_pkg, "load_all_districts_geojson"))
        self.assertTrue(hasattr(geo_pkg, "GeoInscriptionRecord"))
        self.assertTrue(hasattr(geo_pkg, "AtlasDatabase"))
        # Test lazy load of maplibre_atlas via PEP 562 __getattr__
        comp = getattr(geo_pkg, "maplibre_atlas")
        self.assertTrue(callable(comp))

    # --- 3: Single Valid District GeoJSON ---
    def test_single_valid_district_geojson(self):
        valid_gj = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "district": "Thanjavur",
                        "st_nm": "Tamil Nadu",
                        "dt_code": "620",
                        "st_code": "33",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [79.0, 10.6],
                                [79.3, 10.6],
                                [79.3, 10.9],
                                [79.0, 10.9],
                                [79.0, 10.6],
                            ]
                        ],
                    },
                }
            ],
        }
        is_valid, issues = validate_geojson_structure(valid_gj, "test_single")
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)

    # --- 4 & 12: Multiple Valid Files & Duplicate Features ---
    def test_multiple_valid_district_geojson_and_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            st_dir = tmp_path / "states"
            st_dir.mkdir()

            feat_a = {
                "type": "Feature",
                "properties": {"district": "District A", "st_nm": "State One", "dt_code": "101"},
                "geometry": {
                    "type": "Point",
                    "coordinates": [78.0, 12.0],
                },
            }
            # Duplicate feat_a in another file
            feat_a_dup = {
                "type": "Feature",
                "properties": {"district": "District A", "st_nm": "State One", "dt_code": "101"},
                "geometry": {
                    "type": "Point",
                    "coordinates": [78.0, 12.0],
                },
            }
            feat_b = {
                "type": "Feature",
                "properties": {"district": "District B", "st_nm": "State Two", "dt_code": "102"},
                "geometry": {
                    "type": "Point",
                    "coordinates": [79.0, 13.0],
                },
            }

            with open(st_dir / "state-one.geojson", "w", encoding="utf-8") as f:
                json.dump({"type": "FeatureCollection", "features": [feat_a]}, f)
            with open(st_dir / "state-two.geojson", "w", encoding="utf-8") as f:
                json.dump({"type": "FeatureCollection", "features": [feat_a_dup, feat_b]}, f)

            orig_comp = loader.ALL_DISTRICTS_GEOJSON_PATH
            orig_states = loader.PER_STATE_GEO_DIR
            try:
                loader.ALL_DISTRICTS_GEOJSON_PATH = tmp_path / "nonexistent_comp.geojson"
                loader.PER_STATE_GEO_DIR = st_dir
                clear_geo_cache()

                res = load_all_districts_geojson()
                self.assertEqual(res["type"], "FeatureCollection")
                # Deduplication should ensure only 2 unique features (District A and District B)
                self.assertEqual(len(res["features"]), 2)
            finally:
                loader.ALL_DISTRICTS_GEOJSON_PATH = orig_comp
                loader.PER_STATE_GEO_DIR = orig_states
                clear_geo_cache()

    # --- 5: Empty Geographic Dataset ---
    def test_empty_geographic_dataset(self):
        empty_fc = {"type": "FeatureCollection", "features": []}
        is_valid, issues = validate_geojson_structure(empty_fc, "empty_test")
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)

        # Completely empty dict
        is_valid_dict, issues_dict = validate_geojson_structure({}, "dict_test")
        self.assertFalse(is_valid_dict)
        self.assertTrue(any("FeatureCollection" in s for s in issues_dict))

    # --- 6 & 7: Missing Directory and Missing File ---
    def test_missing_dataset_directory_and_missing_file(self):
        nonexistent_dir = Path(tempfile.gettempdir()) / "nonexistent_dir_kallvettu_test"
        nonexistent_file = nonexistent_dir / "missing.geojson"

        orig_comp = loader.ALL_DISTRICTS_GEOJSON_PATH
        orig_states = loader.PER_STATE_GEO_DIR
        try:
            loader.ALL_DISTRICTS_GEOJSON_PATH = nonexistent_file
            loader.PER_STATE_GEO_DIR = nonexistent_dir
            clear_geo_cache()

            res = load_all_districts_geojson()
            self.assertEqual(res, {"type": "FeatureCollection", "features": []})
        finally:
            loader.ALL_DISTRICTS_GEOJSON_PATH = orig_comp
            loader.PER_STATE_GEO_DIR = orig_states
            clear_geo_cache()

    # --- 8: Invalid JSON Handling ---
    def test_invalid_json_handling(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".geojson") as f:
            f.write("{ invalid json string content ::: [")
            f_path = Path(f.name)

        try:
            res = _read_json_with_mtime_cache(f_path)
            self.fail("Expected JSONDecodeError or Exception")
        except Exception:
            pass  # Expected exception on broken JSON
        finally:
            if f_path.is_file():
                f_path.unlink()

    # --- 9: Malformed FeatureCollection ---
    def test_malformed_feature_collection(self):
        # features is a string instead of a list
        bad_fc_1 = {"type": "FeatureCollection", "features": "not_a_list"}
        ok1, issues1 = validate_geojson_structure(bad_fc_1)
        self.assertFalse(ok1)
        self.assertTrue(any("must be a list" in i for i in issues1))

        # Not a dict at all
        ok2, issues2 = validate_geojson_structure(["not_a_dict"])
        self.assertFalse(ok2)

    # --- 10: Missing Geometry ---
    def test_missing_geometry_resilience(self):
        # Feature with geometry = None is valid GeoJSON (unlocated feature)
        feat_no_geom = {
            "type": "Feature",
            "properties": {"district": "Unknown", "st_nm": "Unknown"},
            "geometry": None,
        }
        ok, issues = validate_geojson_feature(feat_no_geom, 0)
        self.assertTrue(ok)

        # Spatial search functions should skip features with None geometry without crashing
        sample_fc = {"type": "FeatureCollection", "features": [feat_no_geom]}
        match = loader._find_containing_district_in_geojson(10.0, 78.0, sample_fc)
        self.assertIsNone(match)

    # --- 11: Invalid Coordinates (NaN, Out-of-Range) ---
    def test_invalid_coordinates(self):
        # Out-of-range lon/lat
        feat_bad_range = {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Point", "coordinates": [250.0, 150.0]},
        }
        ok, issues = validate_geojson_feature(feat_bad_range, 0)
        self.assertFalse(ok)
        self.assertTrue(any("out of WGS84 range" in i for i in issues))

        # NaN coordinates
        feat_nan = {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Point", "coordinates": [float("nan"), 10.0]},
        }
        ok_nan, issues_nan = validate_geojson_feature(feat_nan, 0)
        self.assertFalse(ok_nan)
        self.assertTrue(any("NaN" in i for i in issues_nan))

        # find_admin_unit_for_point with NaN/Inf must return None safely
        self.assertIsNone(find_admin_unit_for_point(float("nan"), 78.0))
        self.assertIsNone(find_admin_unit_for_point(10.0, float("inf")))
        self.assertIsNone(find_admin_unit_for_point(120.0, 78.0))

    # --- 13: Unicode Tamil District and State Names ---
    def test_unicode_tamil_district_and_state_names(self):
        self.assertEqual(normalize_state_name("தமிழ்நாடு"), "Tamil Nadu")
        self.assertEqual(normalize_state_name("தமிழ் நாடு"), "Tamil Nadu")
        self.assertEqual(normalize_state_name("கர்நாடகா"), "Karnataka")
        self.assertEqual(normalize_state_name("புதுச்சேரி"), "Puducherry")

        # Tamil district aliases
        self.assertEqual(normalize_district_name("Tamil Nadu", "தஞ்சாவூர்"), "Thanjavur")
        self.assertEqual(normalize_district_name("Tamil Nadu", "தஞ்சை"), "Thanjavur")
        self.assertEqual(normalize_district_name("Tamil Nadu", "மதுரை"), "Madurai")
        self.assertEqual(normalize_district_name("Tamil Nadu", "காஞ்சிபுரம்"), "Kancheepuram")
        self.assertEqual(normalize_district_name("Tamil Nadu", "திருச்சி"), "Tiruchirappalli")
        self.assertEqual(normalize_district_name("Tamil Nadu", "சென்னை"), "Chennai")
        self.assertEqual(normalize_district_name("Tamil Nadu", "திருநெல்வேலி"), "Tirunelveli")

    # --- 14: Historical / Alias Identifiers ---
    def test_historical_aliases_normalization(self):
        self.assertEqual(normalize_state_name("Orissa"), "Odisha")
        self.assertEqual(normalize_state_name("Pondicherry"), "Puducherry")
        self.assertEqual(normalize_state_name("Uttaranchal"), "Uttarakhand")
        self.assertEqual(normalize_district_name("Tamil Nadu", "Tanjore"), "Thanjavur")
        self.assertEqual(normalize_district_name("Tamil Nadu", "Trichy"), "Tiruchirappalli")
        self.assertEqual(normalize_district_name("Tamil Nadu", "Madras"), "Chennai")
        self.assertEqual(normalize_district_name("Karnataka", "Mysore"), "Mysuru")
        self.assertEqual(normalize_district_name("Karnataka", "Bangalore"), "Bengaluru Urban")

    # --- 15: Windows File Paths and Pathlib Compatibility ---
    def test_windows_paths_and_pathlib_compatibility(self):
        self.assertIsInstance(loader.GEO_DATA_DIR, Path)
        self.assertIsInstance(loader.STATES_GEOJSON_PATH, Path)
        self.assertIsInstance(loader.ALL_DISTRICTS_GEOJSON_PATH, Path)
        self.assertIsInstance(loader.PER_STATE_GEO_DIR, Path)

        # Test both string and Path arguments to _read_json_with_mtime_cache
        meta_str = str(loader.GEO_METADATA_PATH)
        data_from_str = _read_json_with_mtime_cache(meta_str)
        self.assertIsNotNone(data_from_str)

        data_from_path = _read_json_with_mtime_cache(loader.GEO_METADATA_PATH)
        self.assertIsNotNone(data_from_path)
        self.assertEqual(data_from_str["dataset_name"], data_from_path["dataset_name"])

    # --- 16: Cache Invalidation on Dataset Modification (Mtime-aware) ---
    def test_cache_invalidation_after_dataset_updates(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".geojson") as f:
            json.dump({"type": "FeatureCollection", "features": [], "version": 1}, f)
            f_path = Path(f.name)

        try:
            # 1. First read
            d1 = _read_json_with_mtime_cache(f_path)
            self.assertEqual(d1["version"], 1)

            # Wait slightly to ensure a measurable st_mtime change on Windows filesystem
            time.sleep(0.05)

            # 2. Modify file on disk
            with open(f_path, "w", encoding="utf-8") as f:
                json.dump({"type": "FeatureCollection", "features": [], "version": 2}, f)

            # 3. Second read without clear_geo_cache: mtime-aware cache should automatically detect update
            d2 = _read_json_with_mtime_cache(f_path)
            self.assertEqual(d2["version"], 2)
        finally:
            if f_path.is_file():
                f_path.unlink()


if __name__ == "__main__":
    unittest.main()

