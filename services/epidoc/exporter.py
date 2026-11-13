"""
EpiDoc TEI XML Exporter for Ancient Tamil Inscriptions (Phase C).

Follows EpiDoc Guidelines & TEI P5 specification:
- Generates structured XML with <teiHeader>, <msDesc>, <supportDesc>, <handDesc>.
- Parses inscriptional lacunae [ ... ] into <supplied reason="lost"> or <gap>.
- Parses alternative readings [c1/c2] into <app><lem>c1</lem><rdg>c2</rdg></app>.
- Encodes diplomatic and normalized editions with bilingual translations and bibliographic citations.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, Any, Optional, List


def parse_epigraphical_text_to_epidoc(text: str) -> str:
    """
    Transforms plain epigraphical transcription containing standard editorial conventions
    into EpiDoc XML inline tags:
    - [word] -> <supplied reason="lost">word</supplied>
    - [c1/c2] -> <app><lem>c1</lem><rdg>c2</rdg></app>
    - [---] or [...] -> <gap reason="lost" unit="character" extent="unknown"/>
    - (.) -> <gap reason="lost" quantity="1" unit="character"/>
    - {comment} -> <note>comment</note>
    """
    if not text:
        return ""

    out = text

    # 1. Alternative readings [c1/c2]
    def _alt_sub(match: re.Match) -> str:
        content = match.group(1).strip()
        parts = [p.strip() for p in content.split("/") if p.strip()]
        if len(parts) >= 2:
            lem = parts[0]
            rdgs = "".join(f"<rdg>{p}</rdg>" for p in parts[1:])
            return f"<app><lem>{lem}</lem>{rdgs}</app>"
        return match.group(0)

    out = re.sub(r"\[([^\/\]]+\/[^\]]+)\]", _alt_sub, out)

    # 2. Complete lost gaps [---] or [...]
    out = re.sub(r"\[(?:---|\.\.\.|\.\.\.)\]", '<gap reason="lost" unit="character" extent="unknown"/>', out)

    # 3. Single lost characters (.)
    out = re.sub(r"\(\.\)", '<gap reason="lost" quantity="1" unit="character"/>', out)
    out = re.sub(r"\(\.\.\)", '<gap reason="lost" quantity="2" unit="character"/>', out)

    # 4. Standard supplied lacunae [word]
    out = re.sub(r"\[(.*?)\]", r'<supplied reason="lost">\1</supplied>', out)

    # 5. Editorial notes {note}
    out = re.sub(r"\{(.*?)\}", r"<note>\1</note>", out)

    return out


def generate_epidoc_xml(
    inscription_id: str,
    title_en: str,
    title_ta: str,
    script_category: str,
    historical_period: str,
    site_name: str,
    district: str,
    state: str = "Tamil Nadu",
    material_type: str = "Granite Stone Bed",
    century: str = "",
    ruler_name: str = "",
    dynasty: str = "",
    regnal_year: str = "",
    description: str = "",
    diplomatic_text: str = "",
    normalized_text: str = "",
    english_translation: str = "",
    tamil_translation: str = "",
    historical_commentary: str = "",
    citation: str = "",
) -> str:
    """
    Builds a fully compliant, standardized EpiDoc TEI XML string.
    """
    epidoc_diplomatic = parse_epigraphical_text_to_epidoc(diplomatic_text)
    epidoc_normalized = parse_epigraphical_text_to_epidoc(normalized_text)

    # Clean XML-sensitive characters from free-text fields
    def _escape(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    clean_title_en = _escape(title_en or inscription_id)
    clean_title_ta = _escape(title_ta or "")
    clean_site = _escape(site_name or "Unknown Archaeological Site")
    clean_district = _escape(district or "")
    clean_state = _escape(state or "Tamil Nadu")
    clean_material = _escape(material_type or "Granite")
    clean_script = _escape(script_category or "Tamil-Brahmi")
    clean_period = _escape(historical_period or "")
    clean_century = _escape(century or "")
    clean_ruler = _escape(ruler_name or "")
    clean_dynasty = _escape(dynasty or "")
    clean_regnal = _escape(regnal_year or "")
    clean_desc = _escape(description or "")
    clean_en_trans = _escape(english_translation or "")
    clean_ta_trans = _escape(tamil_translation or "")
    clean_commentary = _escape(historical_commentary or clean_desc)
    clean_citation = _escape(citation or "Epigraphia Indica / Archaeological Survey of India")

    ruler_block = ""
    if clean_ruler:
        ruler_block = f"""
            <name type="ruler">{clean_ruler}</name>
            <name type="dynasty">{clean_dynasty}</name>
            <date type="regnal">{clean_regnal}</date>
        """

    xml_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<?xml-model href="http://www.stoa.org/epidoc/schema/latest/tei-epidoc.rng" schematypens="http://relaxng.org/ns/structure/1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:lang="en">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>{clean_title_en} / {clean_title_ta}</title>
        <editor role="creator">Kallvettu Epigraphical Intelligence Platform</editor>
      </titleStmt>
      <publicationStmt>
        <authority>Kallvettu Digital Epigraphy Project</authority>
        <idno type="URI">https://kallvettu.org/inscriptions/{inscription_id}</idno>
        <idno type="local">{inscription_id}</idno>
        <availability>
          <licence target="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International</licence>
        </availability>
      </publicationStmt>
      <sourceDesc>
        <msDesc>
          <msIdentifier>
            <idno>{inscription_id}</idno>
            <repository>{clean_site}</repository>
          </msIdentifier>
          <physDesc>
            <objectDesc>
              <supportDesc>
                <support>
                  <material>{clean_material}</material>
                  <objectType>{clean_material}</objectType>
                </support>
                <layoutDesc>
                  <layout>Lithic inscription engraved on ancient stone surface</layout>
                </layoutDesc>
              </supportDesc>
            </objectDesc>
            <handDesc>
              <handNote scriptName="{clean_script}">{clean_script} of {clean_period}</handNote>
            </handDesc>
          </physDesc>
          <history>
            <origin>
              <origPlace>
                <placeName type="archaeological_site">{clean_site}</placeName>
                <placeName type="district">{clean_district}</placeName>
                <placeName type="state">{clean_state}</placeName>
              </origPlace>
              <origDate notBefore="{clean_century}" notAfter="{clean_century}">{clean_century} ({clean_period})</origDate>
              {ruler_block}
            </origin>
          </history>
        </msDesc>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <langUsage>
        <language ident="tam-Brah">Ancient Tamil in Tamil-Brahmi script</language>
        <language ident="tam-Vatt">Classical Tamil in Vatteluttu script</language>
        <language ident="tam">Classical Inscriptional Tamil</language>
        <language ident="en">English</language>
      </langUsage>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div type="edition" subtype="diplomatic" xml:lang="tam" xml:space="preserve">
        <ab>
          {epidoc_diplomatic}
        </ab>
      </div>
      <div type="edition" subtype="normalized" xml:lang="tam" xml:space="preserve">
        <ab>
          {epidoc_normalized}
        </ab>
      </div>
      <div type="translation" xml:lang="en">
        <p>{clean_en_trans}</p>
      </div>
      <div type="translation" xml:lang="ta">
        <p>{clean_ta_trans}</p>
      </div>
      <div type="commentary">
        <p>{clean_commentary}</p>
      </div>
      <div type="bibliography">
        <listBibl>
          <bibl>{clean_citation}</bibl>
        </listBibl>
      </div>
    </body>
  </text>
</TEI>
"""
    return xml_template


class EpiDocExporter:
    """Class wrapper for EpiDoc export utilities."""

    @staticmethod
    def generate_tei_epidoc_xml(*args, **kwargs) -> str:
        if "modern_tamil_translation" in kwargs:
            kwargs["tamil_translation"] = kwargs.pop("modern_tamil_translation")
        if "citation_source" in kwargs:
            kwargs["citation"] = kwargs.pop("citation_source")
        return generate_epidoc_xml(*args, **kwargs)
