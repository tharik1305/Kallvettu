"""
IIIF Image & Presentation 3.0 Deep-Zoom Viewer Package (Phase D).
"""

from services.iiif.manifest_builder import (
    build_iiif_image_info,
    build_iiif_presentation_manifest,
    IIIFManifestBuilder,
)
from services.iiif.coordinate_mapper import (
    scale_box_to_original,
    scale_box_to_display,
    parse_iiif_region,
)
from services.iiif.viewer_component import (
    render_openseadragon_viewer,
    image_to_base64_data_uri,
)

__all__ = [
    "build_iiif_image_info",
    "build_iiif_presentation_manifest",
    "scale_box_to_original",
    "scale_box_to_display",
    "parse_iiif_region",
    "render_openseadragon_viewer",
    "image_to_base64_data_uri",
]
