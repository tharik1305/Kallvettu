"""
Reflectance Transformation Imaging (RTI) & Dynamic Relighting Engine (Phase J).

Simulates raking light and specular surface enhancement on ancient weathered stone:
- Derives surface normal vector fields (Nx, Ny, Nz) from inscription squeeze/photographs.
- Performs real-time interactive relighting parameterized by virtual light azimuth and elevation.
- Implements Specular Enhancement and Diffuse Gain to expose faint, eroded chisel grooves (pulli, kombu).
"""

from __future__ import annotations

import math
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


class RTIEngine:
    """Simulates RTI polynomial texture mapping and photometric relighting for stone epigraphy."""

    @classmethod
    def compute_surface_normals(
        cls,
        image_bgr: np.ndarray,
        smooth_sigma: float = 1.5,
        depth_scale: float = 2.0,
    ) -> np.ndarray:
        """
        Derives normalized surface normal vector field (H, W, 3) from intensity variations.
        Nx points right, Ny points down, Nz points outward toward the viewer.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if smooth_sigma > 0:
            gray = cv2.GaussianBlur(gray, (0, 0), smooth_sigma)

        # Sobel gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # Normal vector derivation: N = (-gx * scale, -gy * scale, 1.0)
        nx = -gx * (depth_scale / 255.0)
        ny = -gy * (depth_scale / 255.0)
        nz = np.ones_like(nx, dtype=np.float32)

        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        norm = np.maximum(norm, 1e-6)

        normals = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
        return normals

    @classmethod
    def relight_surface(
        cls,
        image_bgr: np.ndarray,
        normals: np.ndarray,
        azimuth_deg: float = 45.0,
        elevation_deg: float = 35.0,
        diffuse_gain: float = 1.0,
        specular_gain: float = 0.5,
        specular_shininess: float = 32.0,
    ) -> np.ndarray:
        """
        Relights the stone surface with a virtual directional light source.
        azimuth_deg: 0 to 360 (angle in XY plane)
        elevation_deg: 5 to 85 (angle above surface plane)
        """
        az_rad = math.radians(azimuth_deg)
        el_rad = math.radians(elevation_deg)

        # Light direction vector pointing from surface to light source
        lx = math.cos(el_rad) * math.cos(az_rad)
        ly = math.cos(el_rad) * math.sin(az_rad)
        lz = math.sin(el_rad)
        light_vec = np.array([lx, ly, lz], dtype=np.float32)

        # Lambertian diffuse: N . L
        diffuse = np.sum(normals * light_vec, axis=-1)
        diffuse = np.clip(diffuse, 0.0, 1.0)

        # Specular component (Blinn-Phong)
        # Viewer vector V = (0, 0, 1)
        view_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        half_vec = light_vec + view_vec
        half_vec /= max(1e-6, np.linalg.norm(half_vec))

        n_dot_h = np.sum(normals * half_vec, axis=-1)
        n_dot_h = np.clip(n_dot_h, 0.0, 1.0)
        specular = np.power(n_dot_h, specular_shininess)

        # Modulate original color texture
        base_f = image_bgr.astype(np.float32) / 255.0
        relit = (
            base_f * (diffuse[..., np.newaxis] * diffuse_gain)
            + (specular[..., np.newaxis] * specular_gain)
        )
        relit_bgr = np.clip(relit * 255.0, 0, 255).astype(np.uint8)
        return relit_bgr

    @classmethod
    def normal_map_to_color(cls, normals: np.ndarray) -> np.ndarray:
        """Encodes normal vector field into RGB color representation for visual inspection."""
        # Convert [-1, 1] to [0, 255]
        norm_rgb = ((normals + 1.0) * 0.5 * 255.0).astype(np.uint8)
        return norm_rgb

