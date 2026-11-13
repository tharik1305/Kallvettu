"""
Interactive Graph Renderer for Epigraphical Knowledge Graph (Phase G).

Generates self-contained, theme-aligned HTML visualizer using Vis.js Network.
Features:
- Dark Graphite (#171D24 / #202832) & Muted Bronze (#C5A15A) aesthetic.
- Category filtering & live search.
- Interactive side panel for citation details, dates, and historical descriptions.
- Responsive physics simulation with node pinning.
"""

from __future__ import annotations

import json
from typing import Dict, Any, Optional
from services.knowledge_graph.graph_engine import EpigraphicalKnowledgeGraph


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Kallvettu Epigraphical Knowledge Graph</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    body {
      background-color: #171D24;
      color: #E2E8F0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      height: __HEIGHT_PX__px;
    }
    #toolbar {
      background-color: #202832;
      padding: 10px 16px;
      display: flex;
      gap: 12px;
      align-items: center;
      border-bottom: 1px solid #2D3748;
      font-size: 13px;
      flex-wrap: wrap;
    }
    .tool-group {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    label {
      color: #C5A15A;
      font-weight: 600;
    }
    input, select {
      background-color: #12171D;
      color: #E2E8F0;
      border: 1px solid #4A5568;
      border-radius: 4px;
      padding: 4px 8px;
      font-size: 13px;
      outline: none;
    }
    input:focus, select:focus {
      border-color: #C5A15A;
    }
    button {
      background-color: #2D3748;
      color: #C5A15A;
      border: 1px solid #C5A15A;
      border-radius: 4px;
      padding: 4px 10px;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    button:hover {
      background-color: #C5A15A;
      color: #12171D;
    }
    #main-container {
      display: flex;
      flex: 1;
      position: relative;
      overflow: hidden;
    }
    #network-canvas {
      flex: 1;
      height: 100%;
      background: radial-gradient(circle at center, #1B232D 0%, #12171E 100%);
    }
    #side-panel {
      width: 320px;
      background-color: #1B232C;
      border-left: 1px solid #2D3748;
      padding: 16px;
      overflow-y: auto;
      font-size: 13px;
      line-height: 1.5;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .panel-header {
      font-size: 16px;
      font-weight: bold;
      color: #C5A15A;
      border-bottom: 1px solid #3B4758;
      padding-bottom: 8px;
    }
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      background-color: #2D3748;
      color: #C5A15A;
    }
    .prop-row {
      margin-bottom: 6px;
    }
    .prop-name {
      color: #A0AEC0;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .prop-val {
      color: #F7FAFC;
      font-size: 13px;
    }
    .citation-box {
      background-color: #12171D;
      border-left: 3px solid #C5A15A;
      padding: 8px;
      border-radius: 0 4px 4px 0;
      font-size: 12px;
      color: #CBD5E0;
    }
    .stat-pill {
      color: #A0AEC0;
      font-size: 11px;
      margin-left: auto;
    }
  </style>
</head>
<body>

  <div id="toolbar">
    <div class="tool-group">
      <label for="categoryFilter">Category:</label>
      <select id="categoryFilter" onchange="filterGraph()">
        <option value="ALL">All Categories</option>
        <option value="RULER">Rulers & Kings</option>
        <option value="DYNASTY">Dynasties</option>
        <option value="TEMPLE">Sacred Sites / Temples</option>
        <option value="LOCATION">Locations</option>
        <option value="DONOR">Donors & Ascetics</option>
        <option value="INSCRIPTION">Inscriptions</option>
      </select>
    </div>

    <div class="tool-group">
      <label for="searchInput">Search:</label>
      <input type="text" id="searchInput" placeholder="Search entity / king..." oninput="filterGraph()">
    </div>

    <button onclick="network.fit({animation: true})">Fit View</button>
    <button onclick="togglePhysics()">Toggle Physics</button>

    <div class="stat-pill">
      <span>__NODE_COUNT__ Entities | __EDGE_COUNT__ Verified Relations</span>
    </div>
  </div>

  <div id="main-container">
    <div id="network-canvas"></div>
    <div id="side-panel">
      <div class="panel-header">Entity Inspection</div>
      <div id="panel-content">
        <p style="color: #718096; font-style: italic;">Select any node in the graph to inspect epigraphical citations, historical dates, and linked records.</p>
      </div>
    </div>
  </div>

  <script type="text/javascript">
    const rawNodes = __NODES_JSON__;
    const rawEdges = __EDGES_JSON__;

    let physicsEnabled = true;

    function buildVisNodes(filterCat, searchTerm) {
      return rawNodes
        .filter(n => {
          const matchCat = filterCat === 'ALL' || n.category === filterCat;
          const matchSearch = !searchTerm ||
            n.label.toLowerCase().includes(searchTerm) ||
            (n.label_ta && n.label_ta.toLowerCase().includes(searchTerm)) ||
            (n.description && n.description.toLowerCase().includes(searchTerm));
          return matchCat && matchSearch;
        })
        .map(n => ({
          id: n.id,
          label: n.label + (n.label_ta ? "\\n(" + n.label_ta + ")" : ""),
          title: n.label,
          color: {
            background: n.color || "#C5A15A",
            border: "#E2E8F0",
            highlight: {
              background: "#FFFFFF",
              border: "#C5A15A"
            }
          },
          font: {
            color: "#FFFFFF",
            size: 13,
            strokeWidth: 2,
            strokeColor: "#12171E"
          },
          shape: n.category === 'INSCRIPTION' ? 'box' : (n.category === 'DYNASTY' ? 'diamond' : 'dot'),
          size: n.category === 'RULER' ? 26 : (n.category === 'DYNASTY' ? 30 : 20),
          raw: n
        }));
    }

    const container = document.getElementById('network-canvas');
    let nodesDataSet = new vis.DataSet(buildVisNodes('ALL', ''));
    let edgesDataSet = new vis.DataSet(rawEdges.map(e => ({
      from: e.source,
      to: e.target,
      label: e.label,
      arrows: 'to',
      color: { color: '#4A5568', highlight: '#C5A15A' },
      font: { color: '#A0AEC0', size: 10, align: 'middle' },
      raw: e
    })));

    const data = {
      nodes: nodesDataSet,
      edges: edgesDataSet
    };

    const options = {
      physics: {
        barnesHut: {
          gravitationalConstant: -3500,
          centralGravity: 0.3,
          springLength: 120,
          springConstant: 0.04,
          damping: 0.09
        },
        stabilization: { iterations: 150 }
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        navigationButtons: true,
        keyboard: true
      }
    };

    const network = new vis.Network(container, data, options);

    network.on('click', function(params) {
      if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        const nodeItem = rawNodes.find(n => n.id === nodeId);
        if (nodeItem) {
          showEntityDetails(nodeItem);
        }
      }
    });

    function showEntityDetails(n) {
      const html = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span class="badge">${n.category}</span>
          <span style="font-size:11px; color:#A0AEC0;">${n.id}</span>
        </div>
        <div style="font-size:15px; font-weight:bold; color:#FFFFFF; margin-top:8px;">${n.label}</div>
        ${n.label_ta ? `<div style="font-size:13px; color:#C5A15A;">${n.label_ta}</div>` : ''}
        
        <div style="margin-top:12px;">
          ${n.period ? `
            <div class="prop-row">
              <div class="prop-name">Historical Era</div>
              <div class="prop-val">${n.period}</div>
            </div>` : ''}
          ${n.century ? `
            <div class="prop-row">
              <div class="prop-name">Century / Chronology</div>
              <div class="prop-val">${n.century}</div>
            </div>` : ''}
          ${n.description ? `
            <div class="prop-row" style="margin-top:8px;">
              <div class="prop-name">Historical Description</div>
              <div class="prop-val" style="color:#CBD5E0;">${n.description}</div>
            </div>` : ''}
          ${n.citation ? `
            <div class="prop-row" style="margin-top:12px;">
              <div class="prop-name">Scholarly Source Citation</div>
              <div class="citation-box">${n.citation}</div>
            </div>` : ''}
        </div>
      `;
      document.getElementById('panel-content').innerHTML = html;
    }

    function filterGraph() {
      const cat = document.getElementById('categoryFilter').value;
      const q = document.getElementById('searchInput').value.trim().toLowerCase();
      const filtered = buildVisNodes(cat, q);
      nodesDataSet.clear();
      nodesDataSet.add(filtered);
    }

    function togglePhysics() {
      physicsEnabled = !physicsEnabled;
      network.setOptions({ physics: { enabled: physicsEnabled } });
    }
  </script>
</body>
</html>
"""


def render_knowledge_graph_html(
    kg: EpigraphicalKnowledgeGraph,
    height_px: int = 650,
) -> str:
    """
    Renders the knowledge graph as an interactive Vis.js visualization.
    """
    graph_data = kg.to_dict()
    nodes_json = json.dumps(graph_data["nodes"], ensure_ascii=False)
    edges_json = json.dumps(graph_data["edges"], ensure_ascii=False)

    return (
        HTML_TEMPLATE.replace("__HEIGHT_PX__", str(height_px))
        .replace("__NODE_COUNT__", str(graph_data["node_count"]))
        .replace("__EDGE_COUNT__", str(graph_data["edge_count"]))
        .replace("__NODES_JSON__", nodes_json)
        .replace("__EDGES_JSON__", edges_json)
    )
