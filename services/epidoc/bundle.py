"""
Multi-File Research Dossier & EpiDoc ZIP Bundle Exporter (Phase C).

Generates the 5 standard scholarly export artifacts:
1. inscription.xml (EpiDoc TEI XML)
2. transcription.txt (Dual-tier inscriptional readings)
3. translation.txt (Modern Tamil & Scholarly English translations)
4. inscription.json (Structured machine-readable research record)
5. research_report.md (Academic epigraphical research dossier)
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Dict, Any, Optional

from services.epidoc.exporter import generate_epidoc_xml
from services.epidoc.validator import validate_epidoc_xml


def create_epidoc_research_bundle(
    inscription_data: Dict[str, Any],
) -> bytes:
    """
    Assembles a complete academic ZIP archive containing all 5 export artifacts.
    """
    insc_id = inscription_data.get("id", "KV-INSC-EXPORT")
    title_en = inscription_data.get("title_en", "")
    title_ta = inscription_data.get("title_ta", "")
    script = inscription_data.get("script_category", "Tamil-Brahmi")
    period = inscription_data.get("historical_period", "")
    site = inscription_data.get("site_name", "")
    district = inscription_data.get("district", "")
    state = inscription_data.get("state", "Tamil Nadu")
    material = inscription_data.get("material_type", "Granite Stone Bed")
    century = inscription_data.get("century", "")
    ruler = inscription_data.get("ruler_name", "")
    dynasty = inscription_data.get("dynasty", "")
    regnal = inscription_data.get("regnal_year", "")
    desc = inscription_data.get("description", "")
    diplomatic = inscription_data.get("diplomatic_text", "")
    normalized = inscription_data.get("normalized_text", "")
    en_trans = inscription_data.get("english_translation", "")
    ta_trans = inscription_data.get("tamil_translation", "")
    commentary = inscription_data.get("historical_commentary", desc)
    citation = inscription_data.get("citation", "Archaeological Survey of India")

    # 1. Generate EpiDoc XML
    xml_content = generate_epidoc_xml(
        inscription_id=insc_id,
        title_en=title_en,
        title_ta=title_ta,
        script_category=script,
        historical_period=period,
        site_name=site,
        district=district,
        state=state,
        material_type=material,
        century=century,
        ruler_name=ruler,
        dynasty=dynasty,
        regnal_year=regnal,
        description=desc,
        diplomatic_text=diplomatic,
        normalized_text=normalized,
        english_translation=en_trans,
        tamil_translation=ta_trans,
        historical_commentary=commentary,
        citation=citation,
    )

    # 2. Generate transcription.txt
    transcription_txt = f"""================================================================================
KALLVETTU EPIGRAPHICAL TRANSCRIPTION DOSSIER
================================================================================
Inscription ID : {insc_id}
Title (English): {title_en}
Title (Tamil)  : {title_ta}
Script         : {script}
Period         : {period}
Site           : {site}, {district}, {state}
Ruler / Dynasty: {ruler} ({dynasty})

--------------------------------------------------------------------------------
TIER 1: DIPLOMATIC TRANSCRIPTION (LITHIC VERBATIM WITH LACUNAE)
--------------------------------------------------------------------------------
{diplomatic}

--------------------------------------------------------------------------------
TIER 2: NORMALIZED EPIGRAPHICAL TRANSCRIPTION (STANDARD TAMIL UNICODE)
--------------------------------------------------------------------------------
{normalized}
================================================================================
"""

    # 3. Generate translation.txt
    translation_txt = f"""================================================================================
KALLVETTU SCHOLARLY TRANSLATION & COMMENTARY
================================================================================
Inscription ID: {insc_id}

--------------------------------------------------------------------------------
1. CONTEMPORARY MODERN TAMIL TRANSLATION (இக்காலத் தமிழ் உரைநடை)
--------------------------------------------------------------------------------
{ta_trans}

--------------------------------------------------------------------------------
2. SCHOLARLY ENGLISH TRANSLATION
--------------------------------------------------------------------------------
{en_trans}

--------------------------------------------------------------------------------
3. HISTORICAL & EPIGRAPHICAL COMMENTARY
--------------------------------------------------------------------------------
{commentary}

Bibliographic Citation:
{citation}
================================================================================
"""

    # 4. Generate inscription.json
    json_content = json.dumps(
        {
            "schema_version": "3.1.0",
            "kallvettu_id": insc_id,
            "metadata": {
                "title_en": title_en,
                "title_ta": title_ta,
                "script_category": script,
                "historical_period": period,
                "site_name": site,
                "district": district,
                "state": state,
                "material_type": material,
                "century": century,
                "ruler_name": ruler,
                "dynasty": dynasty,
                "regnal_year": regnal,
                "citation": citation,
            },
            "transcriptions": {
                "diplomatic": diplomatic,
                "normalized": normalized,
            },
            "translations": {
                "english": en_trans,
                "tamil": ta_trans,
            },
            "commentary": commentary,
            "epidoc_xml_filename": "inscription.xml",
        },
        ensure_ascii=False,
        indent=2,
    )

    # 5. Generate research_report.md
    markdown_report = f"""# 🏛️ Epigraphical Research Dossier: {title_en}

**Local Inscription Title:** {title_ta}  
**Registry Identifier:** `{insc_id}`  
**Palaeographical Script:** {script}  
**Historical Epoch:** {period} ({century})  
**Provenance / Site:** {site}, {district}, {state}  
**Associated Monarch:** {ruler} ({dynasty}, {regnal})  
**Physical Medium:** {material}  

---

## 1. Palaeographic Transcriptions

### Diplomatic Reading (Lithic Verbatim)
```text
{diplomatic}
```

### Normalized Standard Orthography
```text
{normalized}
```

---

## 2. Bilingual Translations

### Contemporary Modern Tamil
> {ta_trans}

### Scholarly English
> {en_trans}

---

## 3. Historical & Epigraphical Commentary
{commentary}

---

## 4. Bibliographic Citation & Archival Source
- {citation}

*Generated by Kallvettu Ancient Tamil Script Intelligence Platform (EpiDoc TEI P5 Compliant).*
"""

    # Package into in-memory ZIP archive
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{insc_id}/inscription.xml", xml_content)
        zf.writestr(f"{insc_id}/transcription.txt", transcription_txt)
        zf.writestr(f"{insc_id}/translation.txt", translation_txt)
        zf.writestr(f"{insc_id}/inscription.json", json_content)
        zf.writestr(f"{insc_id}/research_report.md", markdown_report)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


class EpiDocArchiveBundler:
    """Class wrapper for research bundle creation."""

    @staticmethod
    def create_research_bundle_zip(inscription_id: str = "", **kwargs) -> bytes:
        data = dict(kwargs)
        if inscription_id:
            data["id"] = inscription_id
        return create_epidoc_research_bundle(data)
