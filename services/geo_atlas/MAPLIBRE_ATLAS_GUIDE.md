# Kallvettu Open-Source Geographic Atlas: MapLibre GL JS + OpenFreeMap Guide

## Overview

The **India Epigraphical Geographic Atlas** in Kallvettu provides a Google Maps-like interactive geographic visualization experience using **100% open-source, non-proprietary mapping technologies**.

### Zero Proprietary Vendor Lock-in
- **No Google Maps API** (no credit card requirements, no billing thresholds, no quota limits, no API keys).
- **No Mapbox SDK** (no access tokens or telemetry tracking).
- **No Fabricated Centroids**: Mappable markers strictly require verified archaeological monument or approximate locality coordinates. District-only and regional records contribute to administrative choropleth counts without bogus site pins.

---

## Architecture & Technology Stack

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Application                           │
│                (India Epigraphical Geographic Atlas)                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Bidirectional Events & Props
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               MapLibre GL JS Streamlit Custom Component                │
│                 (services/geo_atlas/maplibre_component)                │
└───────┬───────────────────────────┬────────────────────────────┬───────┘
        │ Vector Tiles              │ GeoJSON Boundaries         │ Pins
        ▼                           ▼                            ▼
┌──────────────────┐    ┌───────────────────────────┐    ┌───────────────┐
│   OpenFreeMap    │    │ Datameet / LGD Boundaries │    │ SQLite Atlas  │
│  Hosted Vector   │    │  (36 States & Districts)  │    │ Inscriptions  │
│      Tiles       │    │      (datasets/geo/)      │    │  (7+ Records) │
│ (OSM Basemap)    │    └───────────────────────────┘    └───────────────┘
└──────────────────┘
```

1. **Map Rendering Engine**: **MapLibre GL JS v4.7.1**
   - Smooth GPU-accelerated client-side vector tile rendering via WebGL.
   - Smooth continuous zooming (0.1 to 18 zoom levels), panning, 3D pitch, and bearing rotation.
   - Client-side marker clustering (`cluster: true`, cluster expansion easing on click).
2. **Vector Basemap**: **OpenFreeMap**
   - Free, open-source vector map tiles powered by OpenStreetMap data.
   - Shows detailed road networks, national highways, cities, settlements, villages, rivers, water bodies, and geographical place names.
   - **Supported Styles**:
     - `Liberty` (Default): `https://tiles.openfreemap.org/styles/liberty`
     - `Bright`: `https://tiles.openfreemap.org/styles/bright`
     - `Positron`: `https://tiles.openfreemap.org/styles/positron`
     - `Dark`: `https://tiles.openfreemap.org/styles/dark`
3. **Administrative Boundaries**: **GeoJSON (Datasets)**
   - 36 Indian States and Union Territories (`datasets/geo/india_states.geojson`).
   - Lazy-loaded per-state District boundary collections (`datasets/geo/states/*.geojson`).
   - District-level choropleth intensity shading based on authenticated inscriptional density.
4. **Data Persistence**: **SQLite Database** (`datasets/kallvettu_atlas.db`)
   - 4 strictly separated provenance contexts: Discovery, Collection, Current Repository, Photograph Capture.
   - Separate verification tiers: `verified`, `approximate`, `district_only`, `unknown`.

---

## Streamlit Integration & Component Lifecycle

The MapLibre component is implemented as a reusable Streamlit component declared via `streamlit.components.v1.declare_component`:

```python
from services.geo_atlas.maplibre_component import maplibre_atlas

event = maplibre_atlas(
    markers_geojson=markers_gj,
    districts_geojson=districts_gj,
    states_geojson=states_gj,
    view_state=view_state,
    style="liberty",
    selected_state="Tamil Nadu",
    selected_district="Madurai",
    selected_inscription_id="TB_MGL_001",
    height=610,
    key="kallvettu_atlas_maplibre"
)
```

### Event Handling & State Synchronization
The frontend communicates directly with Python without page reloads:
- `marker_click` / `inscription_select`: User clicks on an inscription pin; opens the rich popup and updates the detail inspector and workbench.
- `district_select`: User clicks on a district boundary; filters active inscriptions and zooms into district bounds.
- `state_select`: User clicks on a state polygon; drills down into that state's districts.
- `map_click` / `coordinate_pick`: User clicks anywhere on the basemap; captures precise coordinates for manual registration and point-in-district verification.
- `reset_view`: Resets zoom and pan to the national India view.
- `style_change`: Seamlessly swaps vector style without reloading data sources.

### Avoiding Unnecessary Map Reloads
When Streamlit triggers a rerun (e.g. when updating text filters or selecting a dropdown item), `index.html` inspects `if (map)`:
- It **does not destroy or re-create the MapLibre map instance**.
- Instead, it updates the GeoJSON sources via `map.getSource('inscriptions-source').setData(...)` and `map.flyTo(...)`.
- This ensures a flicker-free, 60fps native mapping experience.

---

## Licensing & Attribution

- **Geographical Data**: © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors.
- **Tile Hosting**: Vector tiles hosted by [OpenFreeMap](https://openfreemap.org) under free and open terms.
- **Administrative Boundaries**: Sourced from Datameet / Local Government Directory (LGD), licensed under Open Data Commons Open Database License (ODbL) / Creative Commons Attribution.

