"""
MapLibre GL JS Open-Source Vector Map Component for Streamlit.
Provides interactive OpenFreeMap vector basemap rendering with GeoJSON overlays,
marker clustering, district choropleth, popups, and bidirectional event communication.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, List
import streamlit as st
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Declare Streamlit custom component referencing the index.html bundle
_maplibre_component = components.declare_component(
    "maplibre_atlas",
    path=_COMPONENT_DIR,
)

OPENFREEMAP_STYLES = {
    "liberty": "https://tiles.openfreemap.org/styles/liberty",
    "bright": "https://tiles.openfreemap.org/styles/bright",
    "positron": "https://tiles.openfreemap.org/styles/positron",
    "dark": "https://tiles.openfreemap.org/styles/dark",
}


def maplibre_atlas(
    markers_geojson: Dict[str, Any],
    districts_geojson: Optional[Dict[str, Any]] = None,
    states_geojson: Optional[Dict[str, Any]] = None,
    view_state: Optional[Dict[str, Any]] = None,
    style: str = "liberty",
    selected_state: Optional[str] = None,
    selected_district: Optional[str] = None,
    selected_inscription_id: Optional[str] = None,
    height: int = 620,
    key: Optional[str] = "kallvettu_maplibre_atlas",
    default: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Renders an interactive OpenFreeMap + MapLibre GL JS vector map of India.

    Parameters:
      markers_geojson: GeoJSON FeatureCollection of verified/approximate site pins
      districts_geojson: GeoJSON FeatureCollection of district boundaries with counts
      states_geojson: GeoJSON FeatureCollection of India States and Union Territories
      view_state: Dict with {"center": [longitude, latitude], "zoom": float}
      style: "liberty" | "bright" | "positron" | "dark"
      selected_state: Current active state name
      selected_district: Current active district name
      selected_inscription_id: Current active inscription ID
      height: Map viewport height in pixels
      key: Unique Streamlit component key

    Returns:
      Dict with event payload (e.g. {"event_type": "marker_click", ...}) or None.
    """
    clean_style = style.lower().strip() if style else "liberty"
    if clean_style not in OPENFREEMAP_STYLES:
        clean_style = "liberty"

    # Default India center if view_state not provided
    safe_view = view_state or {"center": [79.2, 21.8], "zoom": 4.15}

    return _maplibre_component(
        markers_geojson=markers_geojson or {"type": "FeatureCollection", "features": []},
        districts_geojson=districts_geojson or {"type": "FeatureCollection", "features": []},
        states_geojson=states_geojson or {"type": "FeatureCollection", "features": []},
        view_state=safe_view,
        style=clean_style,
        selected_state=selected_state,
        selected_district=selected_district,
        selected_inscription_id=selected_inscription_id,
        height=height,
        key=key,
        default=default,
    )

