"""
IIIF Image API 3.0 & Presentation API 3.0 Manifest Generator (Phase D).

Produces compliant IIIF 3.0 manifests and image service info.json payloads
for digital epigraphical assets.
"""

from __future__ import annotations

import json
from typing import Dict, Any, List, Optional


def build_iiif_image_info(
    image_id: str,
    width: int,
    height: int,
    base_uri: str = "https://kallvettu.org/iiif/image",
    tile_size: int = 512,
) -> Dict[str, Any]:
    """
    Constructs an IIIF Image API 3.0 info.json response.
    Spec: https://iiif.io/api/image/3.0/
    """
    img_uri = f"{base_uri.rstrip('/')}/{image_id}"
    scales = [1, 2, 4, 8, 16]

    sizes = []
    w, h = width, height
    for s in scales[:4]:
        sizes.append({"width": w // s, "height": h // s})

    return {
        "@context": "http://iiif.io/api/image/3/context.json",
        "id": img_uri,
        "type": "ImageService3",
        "protocol": "http://iiif.io/api/image",
        "profile": "level2",
        "width": width,
        "height": height,
        "tiles": [
            {
                "width": tile_size,
                "height": tile_size,
                "scaleFactors": scales,
            }
        ],
        "sizes": sizes,
    }


def build_iiif_presentation_manifest(
    inscription_id: str,
    title_en: str,
    title_ta: str,
    images: List[Dict[str, Any]],
    metadata_fields: Optional[Dict[str, str]] = None,
    base_uri: str = "https://kallvettu.org/iiif",
) -> Dict[str, Any]:
    """
    Constructs an IIIF Presentation API 3.0 Manifest.
    Spec: https://iiif.io/api/presentation/3.0/
    """
    manifest_uri = f"{base_uri.rstrip('/')}/manifest/{inscription_id}"

    metadata_list = []
    if metadata_fields:
        for k, v in metadata_fields.items():
            if v:
                metadata_list.append({
                    "label": {"en": [k.replace("_", " ").title()]},
                    "value": {"none": [str(v)]},
                })

    canvases = []
    for idx, img in enumerate(images):
        asset_id = img.get("id", f"canvas_{idx+1}")
        w = img.get("width", 1600)
        h = img.get("height", 1200)
        img_url = img.get("url", f"{base_uri}/image/{asset_id}/full/max/0/default.jpg")

        canvas_uri = f"{manifest_uri}/canvas/{asset_id}"
        annotation_page_uri = f"{canvas_uri}/page"
        annotation_uri = f"{canvas_uri}/annotation/{asset_id}"

        canvases.append({
            "id": canvas_uri,
            "type": "Canvas",
            "label": {"en": [img.get("label", f"Plate {idx+1}")]},
            "height": h,
            "width": w,
            "items": [
                {
                    "id": annotation_page_uri,
                    "type": "AnnotationPage",
                    "items": [
                        {
                            "id": annotation_uri,
                            "type": "Annotation",
                            "motivation": "painting",
                            "body": {
                                "id": img_url,
                                "type": "Image",
                                "format": img.get("mime_type", "image/jpeg"),
                                "width": w,
                                "height": h,
                                "service": [
                                    {
                                        "id": f"{base_uri}/image/{asset_id}",
                                        "type": "ImageService3",
                                        "profile": "level2",
                                    }
                                ],
                            },
                            "target": canvas_uri,
                        }
                    ],
                }
            ],
        })

    return {
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "id": manifest_uri,
        "type": "Manifest",
        "label": {
            "en": [title_en or inscription_id],
            "ta": [title_ta or title_en],
        },
        "metadata": metadata_list,
        "summary": {
            "en": [f"Digital epigraphical manifest for ancient Tamil inscription {inscription_id}."]
        },
        "items": canvases,
    }


class IIIFManifestBuilder:
    """Class wrapper for building IIIF Presentation 3.0 manifests."""

    @staticmethod
    def build_manifest(
        inscription_id: str,
        label: str,
        summary: str = "",
        image_url: str = "",
        metadata_items: Optional[List[Dict[str, str]]] = None,
        width: int = 2048,
        height: int = 1536,
    ) -> Dict[str, Any]:
        meta_dict = {}
        if metadata_items:
            for item in metadata_items:
                meta_dict[item["label"]] = str(item["value"])
        img_entry = {
            "id": f"IMG-{inscription_id}",
            "width": width,
            "height": height,
            "url": image_url or f"/images/{inscription_id}.jpg",
            "label": "Master High-Resolution Image",
        }
        return build_iiif_presentation_manifest(
            inscription_id=inscription_id,
            title_en=label,
            title_ta=label,
            images=[img_entry],
            metadata_fields=meta_dict,
        )
