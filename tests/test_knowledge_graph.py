"""
Unit Tests for Historical Knowledge Graph Engine & Renderer (Phase G).
"""

import json
import pytest
from services.knowledge_graph.graph_engine import EpigraphicalKnowledgeGraph
from services.knowledge_graph.graph_renderer import render_knowledge_graph_html
from services.database.models import EntityCategory


def test_knowledge_graph_initialization():
    """Verify default graph is created and populated with canonical entities."""
    kg = EpigraphicalKnowledgeGraph()
    data = kg.to_dict()

    assert data["node_count"] >= 15
    assert data["edge_count"] >= 12

    # Check key canonical nodes exist
    assert kg.graph.has_node("DYN-CHOLA")
    assert kg.graph.has_node("RULER-RAJARAJA-1")
    assert kg.graph.has_node("SITE-BRIHADISVARA")
    assert kg.graph.has_node("KV-INSC-002")


def test_node_attributes_and_citations():
    """Verify nodes preserve scholarly citations and bilingual titles."""
    kg = EpigraphicalKnowledgeGraph()
    rajaraja = kg.get_entity("RULER-RAJARAJA-1")

    assert rajaraja is not None
    assert rajaraja["category"] == EntityCategory.RULER.value
    assert "முதலாம் இராஜராஜ" in rajaraja["label_ta"]
    assert "South Indian Inscriptions" in rajaraja["source_citation"]

    mangulam = kg.get_entity("KV-INSC-001")
    assert mangulam is not None
    assert "Early Tamil Epigraphy" in mangulam["source_citation"]


def test_pathfinding_and_traversal():
    """Verify historical pathfinding between connected entities."""
    kg = EpigraphicalKnowledgeGraph()

    # Path from Kundavai to Brihadisvara Temple
    path = kg.find_path("DONOR-KUNDAVAI", "SITE-BRIHADISVARA")
    assert len(path) >= 2
    assert path[0] == "DONOR-KUNDAVAI"
    assert path[-1] == "SITE-BRIHADISVARA"

    # Neighbors of Rajaraja Chola I
    neighbors = kg.get_neighbors("RULER-RAJARAJA-1", direction="all")
    assert len(neighbors) >= 2
    neighbor_ids = [n["neighbor_id"] for n in neighbors]
    assert "DYN-CHOLA" in neighbor_ids
    assert "SITE-BRIHADISVARA" in neighbor_ids


def test_subgraph_extraction():
    """Verify filtering subgraph by seed nodes and categories."""
    kg = EpigraphicalKnowledgeGraph()

    # Extract subgraph around Nedunjeliyan with depth 1
    sub = kg.get_subgraph(entity_ids=["RULER-NEDUNJELIYAN"], max_depth=1)
    sub_data = sub.to_dict()
    assert sub_data["node_count"] >= 2
    assert any(n["id"] == "DYN-PANDYA" for n in sub_data["nodes"])

    # Extract only Dynasties
    dyn_sub = kg.get_subgraph(categories=[EntityCategory.DYNASTY.value])
    dyn_data = dyn_sub.to_dict()
    assert all(n["category"] == EntityCategory.DYNASTY.value for n in dyn_data["nodes"])
    assert dyn_data["node_count"] == 4


def test_graph_export_formats():
    """Verify JSON and GraphML export formats."""
    kg = EpigraphicalKnowledgeGraph()

    # JSON export
    json_str = kg.to_json()
    parsed = json.loads(json_str)
    assert "nodes" in parsed
    assert "edges" in parsed
    assert parsed["node_count"] == len(parsed["nodes"])

    # GraphML export
    graphml_xml = kg.to_graphml()
    assert '<?xml' in graphml_xml or '<graphml' in graphml_xml
    assert 'RULER-RAJARAJA-1' in graphml_xml


def test_graph_html_renderer():
    """Verify HTML template generation for Streamlit embed."""
    kg = EpigraphicalKnowledgeGraph()
    html = render_knowledge_graph_html(kg, height_px=600)

    assert "<!DOCTYPE html>" in html
    assert "vis-network" in html
    assert "RULER-RAJARAJA-1" in html
    assert "Brihadisvara Great Temple" in html
    assert "filterGraph" in html

