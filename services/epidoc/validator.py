"""
EpiDoc TEI XML Schema & Structure Validator (Phase C).

Performs structural and semantic validation on generated EpiDoc XML:
- Validates TEI P5 namespace adherence.
- Enforces mandatory epigraphical headers (<teiHeader>, <msDesc>, <edition>).
- Checks balanced markup and tags.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, Any, List

TEI_NAMESPACE = "http://www.tei-c.org/ns/1.0"
NAMESPACES = {"tei": TEI_NAMESPACE}


def validate_epidoc_xml(xml_content: str) -> Dict[str, Any]:
    """
    Validates an EpiDoc XML string against TEI P5 structural rules.
    Returns:
        dict: {"is_valid": bool, "errors": list, "warnings": list, "elements_found": list}
    """
    errors: List[str] = []
    warnings: List[str] = []
    elements_found: List[str] = []

    if not xml_content or not xml_content.strip():
        return {
            "is_valid": False,
            "errors": ["EpiDoc XML document is empty."],
            "warnings": [],
            "elements_found": [],
        }

    # 1. Parse well-formedness
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        return {
            "is_valid": False,
            "errors": [f"XML ParseError: {str(e)}"],
            "warnings": [],
            "elements_found": [],
        }

    # 2. Check root tag and TEI namespace
    expected_root = f"{{{TEI_NAMESPACE}}}TEI"
    if root.tag != expected_root and root.tag != "TEI":
        errors.append(f"Root element must be <TEI xmlns='{TEI_NAMESPACE}'>, found: {root.tag}")

    # Helper to check element presence
    def _find_elem(path: str) -> bool:
        # Try namespaced path and non-namespaced path
        parts = path.split("/")
        ns_path = "/".join(f"tei:{p}" for p in parts)
        res = root.find(ns_path, NAMESPACES)
        if res is not None:
            return True
        return root.find(path) is not None

    mandatory_paths = [
        ("teiHeader", "teiHeader metadata container"),
        ("teiHeader/fileDesc", "fileDesc bibliographic summary"),
        ("teiHeader/fileDesc/titleStmt", "titleStmt inscription title statement"),
        ("teiHeader/fileDesc/publicationStmt", "publicationStmt authority & licensing"),
        ("teiHeader/fileDesc/sourceDesc", "sourceDesc physical carrier description"),
        ("teiHeader/fileDesc/sourceDesc/msDesc", "msDesc manuscript/lithic identifier"),
        ("text", "text content wrapper"),
        ("text/body", "body text container"),
    ]

    for rel_path, desc in mandatory_paths:
        if _find_elem(rel_path):
            elements_found.append(rel_path)
        else:
            errors.append(f"Missing mandatory TEI element: <{rel_path}> ({desc})")

    # 3. Check for at least one edition div
    editions = root.findall(".//tei:div[@type='edition']", NAMESPACES)
    if not editions:
        editions = [d for d in root.findall(".//div") if d.attrib.get("type") == "edition"]

    if editions:
        elements_found.append("div[@type='edition']")
    else:
        errors.append("Missing <div type='edition'> containing inscriptional transcription.")

    # 4. Warnings on missing recommended sections
    recommended = [
        (".//tei:div[@type='translation']", "div[@type='translation']", "English or Modern Tamil translation"),
        (".//tei:div[@type='bibliography']", "div[@type='bibliography']", "Bibliographical citations"),
    ]
    for xpath, label, purpose in recommended:
        found = root.findall(xpath, NAMESPACES)
        if not found:
            non_ns = xpath.replace("tei:", "")
            found = root.findall(non_ns)
        if found:
            elements_found.append(label)
        else:
            warnings.append(f"Recommended element omitted: <{label}> ({purpose})")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "elements_found": elements_found,
    }

