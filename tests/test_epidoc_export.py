"""
Tests for Phase C: EpiDoc-Compliant TEI XML Export Engine.
"""

import io
import zipfile
import pytest

from services.epidoc.exporter import generate_epidoc_xml, parse_epigraphical_text_to_epidoc
from services.epidoc.validator import validate_epidoc_xml
from services.epidoc.bundle import create_epidoc_research_bundle


def test_epigraphical_text_parsing():
    # 1. Supplied lost reading
    res = parse_epigraphical_text_to_epidoc("நாம் [குடுத்த]னவும்")
    assert '<supplied reason="lost">குடுத்த</supplied>' in res

    # 2. Alternative readings
    res_alt = parse_epigraphical_text_to_epidoc("சாசனம் [க/ச]ல்வெட்டு")
    assert "<app><lem>க</lem><rdg>ச</rdg></app>" in res_alt

    # 3. Gaps and lost characters
    res_gap = parse_epigraphical_text_to_epidoc("பாறை (.) [---]")
    assert '<gap reason="lost" quantity="1" unit="character"/>' in res_gap
    assert '<gap reason="lost" unit="character" extent="unknown"/>' in res_gap


def test_generate_and_validate_epidoc_xml():
    xml_str = generate_epidoc_xml(
        inscription_id="KV-INSC-002",
        title_en="Thanjavur Brihadisvara Inscription",
        title_ta="தஞ்சைப் பெருவுடையார் கோயில் கற்றளிக் கல்வெட்டு",
        script_category="Medieval Chola Tamil",
        historical_period="Imperial Chola (c. 1010 CE)",
        site_name="Brihadisvara Temple, Thanjavur",
        district="Thanjavur",
        state="Tamil Nadu",
        material_type="Granite Stone Plinth",
        century="11th Century CE",
        ruler_name="Rajaraja Chola I",
        dynasty="Chola Dynasty",
        regnal_year="26th Regnal Year",
        description="Imperial royal edict on temple donations.",
        diplomatic_text="நாம் குடுத்தனவும் நம் அக்கன் [குடுத்த]னவும் இத்திருக் கற்றளியிலே [கல்லி]லே வெட்டுக",
        normalized_text="நாம் கொடுத்தனவும் நம் அக்கன் கொடுத்தனவும் இத்திருக்கற்றளியிலே கல்லிலே வெட்டுக என்று திருவாய் மொழிந்தருள",
        english_translation="The gifts endowed by Us, by Our elder sister Kundavai, and others—let them be engraved on this stone temple.",
        tamil_translation="மன்னர் இராஜராஜ சோழன் அளித்த கொடைகள் அனைத்தும் கற்றளியின் கல்லிலே பொறிக்கப்பட வேண்டும்.",
        citation="South Indian Inscriptions (SII), Vol. II, No. 1.",
    )

    assert "<?xml version=" in xml_str
    assert "<TEI" in xml_str
    assert "KV-INSC-002" in xml_str
    assert "<supplied reason=\"lost\">குடுத்த</supplied>" in xml_str

    # Validate structural compliance
    validation = validate_epidoc_xml(xml_str)
    assert validation["is_valid"] is True
    assert len(validation["errors"]) == 0
    assert "teiHeader" in validation["elements_found"]
    assert "div[@type='edition']" in validation["elements_found"]


def test_epidoc_research_bundle_zip():
    data = {
        "id": "KV-INSC-001",
        "title_en": "Mangulam Inscription",
        "title_ta": "மாங்குளம் கல்வெட்டு",
        "script_category": "Tamil-Brahmi",
        "historical_period": "Early Sangam (c. 3rd c. BCE)",
        "site_name": "Mangulam",
        "district": "Madurai",
        "state": "Tamil Nadu",
        "diplomatic_text": "கணிய் நந்தஸிரிகுவன் கள்வெள் அறைய் இத்த பாளிய்",
        "normalized_text": "கணியன் நந்தாசிரியருக்கு நெல்வேலி அறையன் அளித்த பள்ளிப்படுக்கை",
        "english_translation": "Rock bed gifted to monk Nanta-asiriyan.",
        "tamil_translation": "நந்தாசிரியருக்கு அமைத்துக் கொடுத்த குகைப் பள்ளிப் படுக்கை.",
        "citation": "Early Tamil Epigraphy, No. 1.",
    }

    zip_bytes = create_epidoc_research_bundle(data)
    assert len(zip_bytes) > 500

    # Inspect zip contents
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        namelist = zf.namelist()
        assert "KV-INSC-001/inscription.xml" in namelist
        assert "KV-INSC-001/transcription.txt" in namelist
        assert "KV-INSC-001/translation.txt" in namelist
        assert "KV-INSC-001/inscription.json" in namelist
        assert "KV-INSC-001/research_report.md" in namelist

        # Validate zipped xml
        xml_in_zip = zf.read("KV-INSC-001/inscription.xml").decode("utf-8")
        val = validate_epidoc_xml(xml_in_zip)
        assert val["is_valid"] is True

