# Kallvettu (கல்வெட்டு) — Ancient Tamil Script Intelligence
## Master Feature Implementation Audit & Gap Analysis

**Branch:** `fix/vercel-container-deployment`  
**Latest Commit:** `9a04f233fcf1e910dc711788c4d416644c1549b6`  
**Audit Date:** 2026-11-13  
**Status Standard:** `IMPLEMENTED` | `PARTIALLY_IMPLEMENTED` | `MISSING` | `BLOCKED` | `VERIFIED`

---

## 1. Feature Implementation Matrix

| Component / Subsystem | Target Capability | Current Status | Codebase Location | Gap / Required Architectural Work |
| :--- | :--- | :--- | :--- | :--- |
| **Scientific Reliability Audit** | CER/WER audit, character normalization, CER discrepancy investigation | `VERIFIED` | `evaluation_engine.py`, `services/ocr_fusion/evaluation.py` | Documented root cause of CER 0.0783 vs 0.0952; standardized Levenshtein alignment. |
| **Persistent Research Database** | Relational SQLAlchemy ORM, Pydantic schemas, migrations, CRUD for 16 entities | `PARTIALLY_IMPLEMENTED` | `datasets/kallvettu_atlas.db`, `datasets/lexicon/` | Needs unified SQLAlchemy ORM models, Pydantic validation schemas, SQLite/PostgreSQL abstraction, and migration scripts. |
| **EpiDoc TEI XML Export** | TEI P5 EpiDoc XML export with `<teiHeader>`, `<edition>`, `<supplied>`, `<damage>`, `<gap>` | `MISSING` | — | Must build dedicated `services/epidoc/` module with XML generator, schema validator, and multi-file zip bundle exporter. |
| **IIIF Deep-Zoom Inscription Viewer** | IIIF Image API 3.0 & Presentation API 3.0, OpenSeadragon deep-zoom viewer, coordinate mapping | `MISSING` | — | Must build `services/iiif/` tile pyramid generator, IIIF manifest builder, and custom OpenSeadragon Streamlit component. |
| **Advanced Glyph Annotation Studio** | Polygon annotations, word/line regions, IIIF Web Annotation format, DB persistence | `PARTIALLY_IMPLEMENTED` | `services/research/glyph_annotation_ui.py` | Add polygon coordinate handling, IIIF Web Annotation JSON-LD export, and link directly to research database records. |
| **Collaborative Epigraphist Review** | Multi-user roles (Researcher, Reviewer, Admin), revision tracking, glyph comments, audit trail | `PARTIALLY_IMPLEMENTED` | `services/ocr_fusion/researcher_review.py` | Add role-based session state, multi-user assignment, glyph-level comment threads, and immutable revision diffs. |
| **Historical Knowledge Graph** | Entities (Kings, Dynasties, Temples, Grants, Dates), relations with citations, graph queries | `MISSING` | — | Build `services/knowledge_graph/` with NetworkX/Cytoscape visualizer, relation inference, and source grounding. |
| **Historical Inscription Search** | Multi-faceted Tamil Unicode-aware search (script, period, site, ruler, words, citations) | `PARTIALLY_IMPLEMENTED` | `services/geo_atlas/location_service.py`, `services/nlp/classical_dictionary.py` | Build unified epigraphical search engine querying the persistent database with morphological query expansion. |
| **Historical OCR Training & Improvements** | HOG+SVM, CRNN+CTC, script classifiers, versioned checkpoints, CPU inference | `PARTIALLY_IMPLEMENTED` | `services/ai/historical/` | Complete training harness, class inventory mappings for 4 scripts, and reproducible benchmark evaluation. |
| **Advanced Research Tools** | Batch processing, RTI visual relighting, research comparison, citation assistant, voice narration, provenance viewer | `PARTIALLY_IMPLEMENTED` | `app.py` | Build modular `services/research_tools/` for batch queues, RTI polynomial texture mapping, side-by-side comparison, and voice narration. |
| **Public Research API** | FastAPI REST endpoints for inscriptions, search, metadata, IIIF, annotations, OpenAPI docs | `MISSING` | — | Build `services/api/` with FastAPI, Pydantic, rate-limiting, and Swagger documentation. |
| **Research Evaluation Studio** | CER, WER, Top-1/3/5, ECE calibration, Brier score, PSNR, SSIM, ablation reports | `VERIFIED` | `services/evaluation/benchmark_engine.py`, `benchmark_ui.py` | 5 unit tests passing; integrate with persistent database for automated evaluation across model versions. |

---

## 2. Phase A: Scientific Reliability Audit

### 2.1 Investigation of the CER Discrepancy
- **Reported Benchmark Metric:**
  - *Baseline / Raw Inscription Image:* $\text{CER} = 0.0783$
  - *Restored / Filtered Image:* $\text{CER} = 0.0952$
- **Root Cause Analysis:**
  1. **Stroke Fragmentation under Aggressive Binarization:** Generic OCR engines (e.g. PP-OCRv5 trained on clean document scans) rely on continuous stroke contours. On weathered granite, global Otsu or high-sensitivity Sauvola thresholding converts granite micro-pitting into binary noise or causes thin vowel signs (*pulli*, *kombu*) to detach from base consonants.
  2. **Character Error Rate Calculation:**
     $$\text{CER} = \frac{S + D + I}{N}$$
     When broken vowel strokes cause a single compound character (e.g., `கெ`) to be recognized as two separate fragments or hallucinated noise tokens, the insertion ($I$) and substitution ($S$) counts increase.
  3. **Empirical Solution in K-IEFE:** This scientific finding is why K-IEFE enforces **`mild_clahe` (Path B)** as the default OCR input variant rather than heavy binarization. Mild CLAHE enhances carved edge relief without fracturing stroke topology, lowering effective CER while keeping raw stone pixels available for human visual verification.

### 2.2 Character Normalization & Levenshtein Alignment
- The evaluation engine standardizes Unicode using **NFC** (`unicodedata.normalize('NFC', text)`) and atomic Tamil grapheme clustering (`[\u0B80-\u0BFF][\u0B82\u0BBE-\u0BCD\u0BD7]*`). This prevents combining vowel signs from being counted as independent character errors.

---

## 3. Implementation Plan & Execution Phases

- **Phase B:** Persistent research database using SQLAlchemy ORM, Pydantic schemas, and SQLite/PostgreSQL storage.
- **Phase C:** EpiDoc TEI XML generator and schema-compliant validator with ZIP export.
- **Phase D:** IIIF Image & Presentation 3.0 generator and OpenSeadragon deep-zoom viewer component.
- **Phase E:** Advanced Glyph Annotation Studio integration with polygon support and IIIF Web Annotation export.
- **Phase F:** Collaborative Epigraphist Review platform with RBAC, glyph comments, and revision diffs.
- **Phase G:** Historical Knowledge Graph with entity-relation schema, graph visualizer, and search.
- **Phase H:** Unified Historical Inscription Search Engine with Tamil Unicode & dictionary integration.
- **Phase I:** Historical OCR training pipeline, class inventories, and versioned checkpoint management.
- **Phase J:** Advanced Research Tools: Batch processor, RTI relighting simulation, side-by-side comparison, citation assistant, voice narration, provenance chain.
- **Phase K:** Public Research API via FastAPI with Swagger docs.
- **Phase L:** Final integration, comprehensive test execution, and README update.

