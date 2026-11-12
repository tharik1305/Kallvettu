"""
Lightweight CPU-First Historical Glyph Dataset Manager (Phase 1).

Manages the curated historical inscription glyph dataset at:
  datasets/historical_glyphs/
    raw/          - Unmodified source inscription photographs (byte-exact preservation)
    processed/    - Cropped/normalized glyph regions for downstream recognition
    annotations/  - UTF-8 JSONL character-region & uncertainty annotation records
    synthetic/    - Controlled CPU augmentations linked to parent source_id (never leaked)
    splits/       - Reproducible source-inscription-level train/val/test manifests
    metadata/     - Provenance/licensing registry & script character inventories

Strict Data Integrity Rules:
1. Original raw images are stored unmodified and verified via SHA-256.
2. Tamil Unicode labels & alternative readings are NFC-normalized.
3. Splitting operates strictly at the source-inscription level BEFORE any
   synthetic expansion; synthetic variants inherit their parent's split.
"""

import csv
import hashlib
import io
import json
import os
import random
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from PIL import Image

from config import sanitize_filename
from restoration_engine import (
    validate_and_load_image,
    crop_region_of_interest,
    apply_clahe_pillow,
)
from nlp_translation_engine import EPIGRAPHICAL_SCRIPTS

DEFAULT_GLYPH_DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "historical_glyphs")
)

SUPPORTED_HISTORICAL_SCRIPTS = (
    "Tamil-Brahmi",
    "Vatteluttu",
    "Grantha",
    "Medieval Tamil",
)

VALID_UNCERTAINTY_STATUSES = ("intact", "uncertain", "damaged", "illegible")
VALID_APPROVAL_STATUSES = ("pending", "approved", "rejected")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_tamil_unicode(text: Optional[str]) -> str:
    """Normalizes Tamil Unicode text consistently using NFC without altering characters."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", str(text).strip())


def _default_script_inventories() -> Dict[str, List[str]]:
    """Builds initial script-specific Tamil Unicode inventories from EPIGRAPHICAL_SCRIPTS."""
    return {
        "Tamil-Brahmi": [
            "அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ", "ஐ", "ஒ", "ஓ",
            "க", "ங", "ச", "ஞ", "ட", "ண", "த", "ந", "ப", "ம",
            "ய", "ர", "ல", "வ", "ழ", "ள", "ற", "ன", "ஸ",
            "கி", "கு", "கொ", "ண", "ணி", "தா", "தி", "து", "ந்த", "பாளி", "ய்",
        ],
        "Vatteluttu": [
            "அ", "ஆ", "இ", "உ", "எ", "ஒ",
            "க", "ச", "ட", "ண", "த", "ந", "ப", "ம", "ய", "ர", "ல", "வ", "ழ", "ள", "ற", "ன",
            "கோ", "சே", "ந்", "தன்", "கூ", "ற்", "யா", "ண்", "டு", "ளி",
        ],
        "Grantha": [
            "ஸ்ரீ", "ஸ்வ", "ஸ்தி", "தே", "வ", "ர்", "பி", "ர", "ச", "சா", "ன", "ம்",
            "ஶ", "ஷ", "ஸ", "ஹ", "க்ஷ",
        ],
        "Medieval Tamil": [
            "அ", "ஆ", "இ", "உ", "எ", "ஏ", "ஒ", "ஓ",
            "க", "ச", "ஞ", "ட", "ண", "த", "ந", "ப", "ம", "ய", "ர", "ல", "வ", "ழ", "ள", "ற", "ன",
            "நாம்", "கு", "டு", "த்த", "தி", "ரு", "ஸ்ரீ", "யா", "ண்டு",
        ],
    }


@dataclass
class GlyphSourceRecord:
    """Provenance and file integrity record for an unmodified raw inscription image."""
    source_id: str
    original_filename: str
    raw_image_path: str
    sha256: str
    width: int
    height: int
    script: str = "Tamil-Brahmi"
    provenance: str = ""
    license: str = "Research / CC-BY-4.0"
    attribution: str = ""
    notes: str = ""
    created_utc: str = field(default_factory=_now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GlyphSourceRecord":
        return cls(**d)


@dataclass
class GlyphAnnotationRecord:
    """Character-region annotation stored cleanly as JSONL with Tamil Unicode & review status."""
    annotation_id: str
    source_id: str
    parent_source_id: str
    script: str
    character_label: str
    bbox_pixels: List[int]        # [x1, y1, x2, y2] in source coordinates
    bbox_norm: List[float]        # [ymin, xmin, ymax, xmax] in [0.0, 1.0]
    crop_path: str                # Relative path inside dataset root (e.g., processed/...)
    alternative_readings: List[str] = field(default_factory=list)
    uncertainty_status: str = "intact"
    is_damaged: bool = False
    is_illegible: bool = False
    is_synthetic: bool = False
    preprocessing_variant: str = "raw_crop"
    annotation_source: str = "manual_researcher"
    annotator: str = "Epigraphist"
    approval_status: str = "pending"
    approved_by: str = ""
    notes: str = ""
    created_utc: str = field(default_factory=_now_utc)
    updated_utc: str = field(default_factory=_now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GlyphAnnotationRecord":
        data = dict(d)
        data["character_label"] = normalize_tamil_unicode(data.get("character_label", ""))
        data["alternative_readings"] = [
            normalize_tamil_unicode(r) for r in data.get("alternative_readings", []) if normalize_tamil_unicode(r)
        ]
        return cls(**data)


class HistoricalGlyphDatasetManager:
    """
    Manages datasets/historical_glyphs/ storage, annotations, provenance,
    script inventories, source-level train/val/test splits, and leakage checks.
    """

    SUBDIRS = ("raw", "processed", "annotations", "synthetic", "splits", "metadata")

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = os.path.abspath(base_dir or DEFAULT_GLYPH_DATASET_DIR)
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        for sub in self.SUBDIRS:
            os.makedirs(os.path.join(self.base_dir, sub), exist_ok=True)

        inv_path = self.inventories_path
        if not os.path.exists(inv_path):
            with open(inv_path, "w", encoding="utf-8") as f:
                json.dump(_default_script_inventories(), f, ensure_ascii=False, indent=2)

        if not os.path.exists(self.sources_path):
            with open(self.sources_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

        if not os.path.exists(self.annotations_path):
            with open(self.annotations_path, "w", encoding="utf-8") as f:
                f.write("")

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.base_dir, "raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.base_dir, "processed")

    @property
    def synthetic_dir(self) -> str:
        return os.path.join(self.base_dir, "synthetic")

    @property
    def annotations_path(self) -> str:
        return os.path.join(self.base_dir, "annotations", "glyph_annotations.jsonl")

    @property
    def sources_path(self) -> str:
        return os.path.join(self.base_dir, "metadata", "sources.json")

    @property
    def inventories_path(self) -> str:
        return os.path.join(self.base_dir, "metadata", "script_inventories.json")

    @property
    def split_manifest_path(self) -> str:
        return os.path.join(self.base_dir, "splits", "split_manifest.json")

    # ------------------------------------------------------------------
    # Source Image Registration (Unmodified Raw Bytes + SHA-256)
    # ------------------------------------------------------------------

    def load_sources(self) -> List[GlyphSourceRecord]:
        if not os.path.exists(self.sources_path):
            return []
        with open(self.sources_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
        return [GlyphSourceRecord.from_dict(item) for item in raw_list]

    def _save_sources(self, sources: List[GlyphSourceRecord]) -> None:
        with open(self.sources_path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in sources], f, ensure_ascii=False, indent=2)

    def get_source(self, source_id: str) -> Optional[GlyphSourceRecord]:
        for src in self.load_sources():
            if src.source_id == source_id:
                return src
        return None

    def register_source_image(
        self,
        image_bytes: bytes,
        original_filename: str,
        script: str = "Tamil-Brahmi",
        provenance: str = "",
        license_str: str = "Research / CC-BY-4.0",
        attribution: str = "",
        notes: str = "",
    ) -> GlyphSourceRecord:
        """
        Stores raw image bytes UNMODIFIED inside datasets/historical_glyphs/raw/
        and records cryptographic SHA-256 + provenance metadata.
        Deduplicates cleanly by SHA-256 if already registered.
        """
        if not image_bytes:
            raise ValueError("Empty image byte buffer cannot be registered.")

        sha = hashlib.sha256(image_bytes).hexdigest()
        existing = self.load_sources()
        for src in existing:
            if src.sha256 == sha:
                return src

        # Validate image integrity without mutating original_bytes
        pristine_img, _, meta = validate_and_load_image(image_bytes, max_processing_dim=1600)
        safe_name = sanitize_filename(original_filename, default="inscription.png")
        source_id = f"SRC_{sha[:10]}"
        raw_filename = f"{source_id}_{safe_name}"
        raw_abs = os.path.join(self.raw_dir, raw_filename)
        if not raw_abs.startswith(self.raw_dir):
            raise ValueError("Invalid target path for raw image.")

        # Write original bytes verbatim — never re-encoded or overwritten
        with open(raw_abs, "wb") as f:
            f.write(image_bytes)

        rel_raw = os.path.relpath(raw_abs, self.base_dir).replace("\\", "/")
        rec = GlyphSourceRecord(
            source_id=source_id,
            original_filename=safe_name,
            raw_image_path=rel_raw,
            sha256=sha,
            width=int(pristine_img.width),
            height=int(pristine_img.height),
            script=script,
            provenance=provenance.strip(),
            license=license_str.strip(),
            attribution=attribution.strip(),
            notes=notes.strip(),
        )
        existing.append(rec)
        self._save_sources(existing)
        return rec

    def verify_source_integrity(self, source_id: str) -> Tuple[bool, str]:
        """Verifies that the stored raw image still matches its registered SHA-256 hash."""
        src = self.get_source(source_id)
        if not src:
            return False, f"Source '{source_id}' not found."
        abs_path = os.path.join(self.base_dir, src.raw_image_path)
        if not os.path.isfile(abs_path):
            return False, f"Raw file missing: {src.raw_image_path}"
        with open(abs_path, "rb") as f:
            cur_sha = hashlib.sha256(f.read()).hexdigest()
        if cur_sha != src.sha256:
            return False, f"SHA-256 mismatch for '{source_id}' (expected {src.sha256}, got {cur_sha})."
        return True, "Original image verified unchanged."

    # ------------------------------------------------------------------
    # Script Inventories
    # ------------------------------------------------------------------

    def load_script_inventories(self) -> Dict[str, List[str]]:
        if not os.path.exists(self.inventories_path):
            return _default_script_inventories()
        with open(self.inventories_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def add_character_to_inventory(self, script: str, char_label: str) -> None:
        norm_char = normalize_tamil_unicode(char_label)
        if not norm_char:
            return
        inv = self.load_script_inventories()
        lst = inv.setdefault(script, [])
        if norm_char not in lst:
            lst.append(norm_char)
            with open(self.inventories_path, "w", encoding="utf-8") as f:
                json.dump(inv, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Character-Region Annotations (JSONL)
    # ------------------------------------------------------------------

    def load_annotations(
        self,
        script: Optional[str] = None,
        approval_status: Optional[str] = None,
    ) -> List[GlyphAnnotationRecord]:
        if not os.path.exists(self.annotations_path):
            return []
        records: List[GlyphAnnotationRecord] = []
        with open(self.annotations_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = GlyphAnnotationRecord.from_dict(json.loads(line))
                if script and rec.script != script:
                    continue
                if approval_status and rec.approval_status != approval_status:
                    continue
                records.append(rec)
        return records

    def _save_all_annotations(self, records: List[GlyphAnnotationRecord]) -> None:
        with open(self.annotations_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    def annotate_glyph(
        self,
        source_id: str,
        bbox_pixels: Tuple[int, int, int, int],
        character_label: str,
        script: str,
        alternative_readings: Optional[List[str]] = None,
        uncertainty_status: str = "intact",
        is_damaged: bool = False,
        is_illegible: bool = False,
        preprocessing_variant: str = "raw_crop",
        annotation_source: str = "manual_researcher",
        annotator: str = "Epigraphist",
        approval_status: str = "pending",
        approved_by: str = "",
        notes: str = "",
        is_synthetic: bool = False,
        parent_source_id: Optional[str] = None,
    ) -> GlyphAnnotationRecord:
        """
        Crops a glyph region cleanly from the unmodified source image, writes the crop
        to processed/ (or synthetic/), normalizes Tamil Unicode labels, and stores the
        record in annotations/glyph_annotations.jsonl.
        """
        src = self.get_source(source_id)
        if not src:
            raise ValueError(f"Source image '{source_id}' is not registered.")

        norm_label = normalize_tamil_unicode(character_label)
        if not norm_label and not is_illegible:
            raise ValueError("Tamil character label is required unless glyph is marked illegible.")
        if is_illegible and not norm_label:
            norm_label = "[?]"

        if uncertainty_status not in VALID_UNCERTAINTY_STATUSES:
            raise ValueError(f"Invalid uncertainty_status '{uncertainty_status}'.")
        if approval_status not in VALID_APPROVAL_STATUSES:
            raise ValueError(f"Invalid approval_status '{approval_status}'.")

        x1, y1, x2, y2 = [int(v) for v in bbox_pixels]
        raw_abs = os.path.join(self.base_dir, src.raw_image_path)
        with Image.open(raw_abs) as raw_pil:
            rgb_pil = raw_pil.convert("RGB")
            w, h = rgb_pil.size
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            crop_img = crop_region_of_interest(rgb_pil, (x1, y1, x2, y2))

            if preprocessing_variant == "mild_clahe":
                crop_out = apply_clahe_pillow(crop_img, clip_limit=1.8, blend_strength=0.80)
            elif preprocessing_variant == "grayscale":
                crop_out = crop_img.convert("L").convert("RGB")
            else:
                crop_out = crop_img

        all_recs = self.load_annotations()
        ann_idx = len(all_recs) + 1
        ann_id = f"GLY_{ann_idx:05d}_{hashlib.sha1(f'{source_id}_{x1}_{y1}_{x2}_{y2}_{_now_utc()}'.encode()).hexdigest()[:6]}"

        target_subdir = self.synthetic_dir if is_synthetic else self.processed_dir
        crop_filename = sanitize_filename(f"{ann_id}.png")
        crop_abs = os.path.join(target_subdir, crop_filename)
        crop_out.save(crop_abs, format="PNG")
        rel_crop = os.path.relpath(crop_abs, self.base_dir).replace("\\", "/")

        norm_alts = []
        for alt in (alternative_readings or []):
            na = normalize_tamil_unicode(alt)
            if na and na not in norm_alts:
                norm_alts.append(na)

        bbox_norm = [
            round(y1 / float(h), 6),
            round(x1 / float(w), 6),
            round(y2 / float(h), 6),
            round(x2 / float(w), 6),
        ]

        now = _now_utc()
        rec = GlyphAnnotationRecord(
            annotation_id=ann_id,
            source_id=source_id,
            parent_source_id=parent_source_id or source_id,
            script=script,
            character_label=norm_label,
            bbox_pixels=[x1, y1, x2, y2],
            bbox_norm=bbox_norm,
            crop_path=rel_crop,
            alternative_readings=norm_alts,
            uncertainty_status=uncertainty_status,
            is_damaged=bool(is_damaged or uncertainty_status == "damaged"),
            is_illegible=bool(is_illegible or uncertainty_status == "illegible"),
            is_synthetic=bool(is_synthetic),
            preprocessing_variant=preprocessing_variant,
            annotation_source=annotation_source,
            annotator=annotator.strip() or "Epigraphist",
            approval_status=approval_status,
            approved_by=approved_by.strip() if approval_status == "approved" else "",
            notes=notes.strip(),
            created_utc=now,
            updated_utc=now,
        )

        with open(self.annotations_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

        if not rec.is_illegible and norm_label != "[?]":
            self.add_character_to_inventory(script, norm_label)

        return rec

    def update_approval(
        self,
        annotation_id: str,
        approval_status: str,
        approved_by: str = "Reviewer",
    ) -> bool:
        if approval_status not in VALID_APPROVAL_STATUSES:
            raise ValueError(f"Invalid approval_status '{approval_status}'.")
        records = self.load_annotations()
        updated = False
        for rec in records:
            if rec.annotation_id == annotation_id:
                rec.approval_status = approval_status
                rec.approved_by = approved_by.strip()
                rec.updated_utc = _now_utc()
                updated = True
                break
        if updated:
            self._save_all_annotations(records)
        return updated

    # ------------------------------------------------------------------
    # Source-Inscription-Level Splits & Strict Leakage Checks
    # ------------------------------------------------------------------

    def generate_source_level_splits(
        self,
        test_ratio: float = 0.20,
        val_ratio: float = 0.20,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Splits original source inscriptions strictly BEFORE augmentation.
        Any synthetic annotations linked via parent_source_id stay in the same
        partition as their parent source inscription — never split across sets.
        """
        sources = self.load_sources()
        annotations = self.load_annotations()

        source_ids = sorted({s.source_id for s in sources} | {a.parent_source_id for a in annotations})
        if not source_ids:
            manifest = {
                "seed": seed,
                "generated_utc": _now_utc(),
                "source_splits": {"train": [], "val": [], "test": []},
                "annotation_splits": {"train": [], "val": [], "test": []},
                "counts": {"train": 0, "val": 0, "test": 0},
                "leakage_clean": True,
                "leakage_violations": [],
            }
            with open(self.split_manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            return manifest

        rng = random.Random(seed)
        shuffled = list(source_ids)
        rng.shuffle(shuffled)

        n = len(shuffled)
        if n == 1:
            train_ids = list(shuffled)
            val_ids: List[str] = []
            test_ids: List[str] = []
        elif n == 2:
            test_ids = [shuffled[0]]
            train_ids = [shuffled[1]]
            val_ids = []
        else:
            n_test = max(1, int(round(n * test_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            if n_test + n_val >= n:
                n_val = max(0, n - n_test - 1)
            test_ids = sorted(shuffled[:n_test])
            val_ids = sorted(shuffled[n_test:n_test + n_val])
            train_ids = sorted(shuffled[n_test + n_val:])

        split_by_source: Dict[str, str] = {}
        for sid in train_ids:
            split_by_source[sid] = "train"
        for sid in val_ids:
            split_by_source[sid] = "val"
        for sid in test_ids:
            split_by_source[sid] = "test"

        ann_splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        for rec in annotations:
            # Anchor strictly to parent_source_id so synthetic derivatives never leak
            bucket = split_by_source.get(rec.parent_source_id, "train")
            ann_splits[bucket].append(rec.annotation_id)

        manifest = {
            "seed": seed,
            "test_ratio": test_ratio,
            "val_ratio": val_ratio,
            "generated_utc": _now_utc(),
            "source_splits": {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            },
            "annotation_splits": ann_splits,
            "counts": {
                "train": len(ann_splits["train"]),
                "val": len(ann_splits["val"]),
                "test": len(ann_splits["test"]),
            },
        }
        is_clean, violations = self.verify_no_data_leakage(manifest)
        manifest["leakage_clean"] = is_clean
        manifest["leakage_violations"] = violations

        with open(self.split_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        return manifest

    def verify_no_data_leakage(
        self,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Audits train/val/test partitions for source-inscription overlap, SHA-256 overlap,
        or synthetic variant cross-partition contamination.
        """
        if manifest is None:
            if not os.path.exists(self.split_manifest_path):
                return True, []
            with open(self.split_manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        violations: List[str] = []
        src_splits = manifest.get("source_splits", {})
        ann_splits = manifest.get("annotation_splits", {})

        train_s = set(src_splits.get("train", []))
        val_s = set(src_splits.get("val", []))
        test_s = set(src_splits.get("test", []))

        for name_a, set_a, name_b, set_b in [
            ("train", train_s, "val", val_s),
            ("train", train_s, "test", test_s),
            ("val", val_s, "test", test_s),
        ]:
            overlap = set_a & set_b
            if overlap:
                violations.append(f"Source overlap between {name_a} and {name_b}: {sorted(overlap)}")

        # SHA-256 check across source splits
        sources_by_id = {s.source_id: s for s in self.load_sources()}
        sha_map: Dict[str, str] = {}
        for split_name, sids in [("train", train_s), ("val", val_s), ("test", test_s)]:
            for sid in sids:
                src = sources_by_id.get(sid)
                if not src:
                    continue
                prev = sha_map.get(src.sha256)
                if prev and prev != split_name:
                    violations.append(
                        f"Identical image SHA-256 ({src.sha256[:10]}) found in both {prev} and {split_name}"
                    )
                sha_map[src.sha256] = split_name

        # Synthetic parent alignment check
        annotations_by_id = {a.annotation_id: a for a in self.load_annotations()}
        src_to_split = {}
        for s_name, sids in [("train", train_s), ("val", val_s), ("test", test_s)]:
            for sid in sids:
                src_to_split[sid] = s_name

        seen_anns: Dict[str, str] = {}
        for split_name, aids in ann_splits.items():
            for aid in aids:
                if aid in seen_anns and seen_anns[aid] != split_name:
                    violations.append(f"Annotation '{aid}' duplicated across {seen_anns[aid]} and {split_name}")
                seen_anns[aid] = split_name

                rec = annotations_by_id.get(aid)
                if rec:
                    expected_split = src_to_split.get(rec.parent_source_id)
                    if expected_split and expected_split != split_name:
                        violations.append(
                            f"Leakage: annotation '{aid}' (parent_source_id='{rec.parent_source_id}', "
                            f"synthetic={rec.is_synthetic}) placed in '{split_name}' instead of '{expected_split}'"
                        )

        return (len(violations) == 0, violations)

    # ------------------------------------------------------------------
    # Export Utilities (JSON & CSV preserving Tamil Unicode)
    # ------------------------------------------------------------------

    def export_annotations_json(self) -> str:
        payload = {
            "dataset": "Kallvettu Historical Glyph Dataset",
            "exported_utc": _now_utc(),
            "sources": [s.to_dict() for s in self.load_sources()],
            "annotations": [a.to_dict() for a in self.load_annotations()],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def export_annotations_csv(self) -> str:
        annotations = self.load_annotations()
        sources_by_id = {s.source_id: s for s in self.load_sources()}
        buf = io.StringIO()
        fieldnames = [
            "annotation_id", "source_id", "parent_source_id", "script",
            "character_label", "alternative_readings", "uncertainty_status",
            "is_damaged", "is_illegible", "is_synthetic",
            "x1", "y1", "x2", "y2", "ymin", "xmin", "ymax", "xmax",
            "crop_path", "preprocessing_variant", "approval_status",
            "approved_by", "annotator", "provenance", "license", "notes", "created_utc",
        ]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for rec in annotations:
            src = sources_by_id.get(rec.source_id)
            x1, y1, x2, y2 = (rec.bbox_pixels + [0, 0, 0, 0])[:4]
            ymin, xmin, ymax, xmax = (rec.bbox_norm + [0.0, 0.0, 0.0, 0.0])[:4]
            writer.writerow({
                "annotation_id": rec.annotation_id,
                "source_id": rec.source_id,
                "parent_source_id": rec.parent_source_id,
                "script": rec.script,
                "character_label": rec.character_label,
                "alternative_readings": "|".join(rec.alternative_readings),
                "uncertainty_status": rec.uncertainty_status,
                "is_damaged": rec.is_damaged,
                "is_illegible": rec.is_illegible,
                "is_synthetic": rec.is_synthetic,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax,
                "crop_path": rec.crop_path,
                "preprocessing_variant": rec.preprocessing_variant,
                "approval_status": rec.approval_status,
                "approved_by": rec.approved_by,
                "annotator": rec.annotator,
                "provenance": src.provenance if src else "",
                "license": src.license if src else "",
                "notes": rec.notes,
                "created_utc": rec.created_utc,
            })
        return buf.getvalue()

    def get_summary_stats(self) -> Dict[str, Any]:
        sources = self.load_sources()
        anns = self.load_annotations()
        by_script: Dict[str, int] = {}
        approved = 0
        damaged = 0
        for a in anns:
            by_script[a.script] = by_script.get(a.script, 0) + 1
            if a.approval_status == "approved":
                approved += 1
            if a.is_damaged or a.is_illegible:
                damaged += 1
        return {
            "total_sources": len(sources),
            "total_annotations": len(anns),
            "approved_annotations": approved,
            "damaged_or_illegible": damaged,
            "unique_labels": len({a.character_label for a in anns if not a.is_illegible}),
            "by_script": by_script,
        }
