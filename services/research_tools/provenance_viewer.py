"""
Cryptographic Provenance Flow Visualizer (Phase J).

Visualizes the complete evidentiary provenance graph:
Raw Stone Inscription -> Image Capture -> Preprocessing (Mild CLAHE) ->
OCR Extraction -> K-IEFE Consensus Arbitration -> Epigraphist Review -> EpiDoc Export.

Generates dark graphite/muted bronze HTML/SVG diagrams with cryptographic audit hashes.
"""

from __future__ import annotations

import html
from typing import Dict, Any, List, Optional


class ProvenanceViewer:
    """Renders scholarly provenance chain for an epigraphical inscription."""

    @classmethod
    def render_provenance_timeline_html(
        cls,
        inscription_id: str,
        provenance_events: List[Dict[str, Any]],
    ) -> str:
        """
        Renders an HTML timeline of provenance events.
        """
        event_nodes_html = []
        for idx, ev in enumerate(provenance_events):
            title = html.escape(ev.get("stage_name", "Processing Event"))
            agent = html.escape(ev.get("agent_or_provider", "System"))
            timestamp = html.escape(ev.get("timestamp", ""))
            details = html.escape(ev.get("details", ""))
            sha = html.escape(ev.get("sha256", "")[:16] + "...") if ev.get("sha256") else ""

            is_last = (idx == len(provenance_events) - 1)
            line_html = "" if is_last else '<div style="position:absolute; left:19px; top:40px; bottom:-16px; width:2px; background:#2D3748;"></div>'

            event_nodes_html.append(f"""
            <div style="position:relative; display:flex; gap:16px; margin-bottom:20px;">
              <div style="position:relative; width:40px; display:flex; justify-content:center;">
                <div style="width:16px; height:16px; border-radius:50%; background:#C5A15A; border:3px solid #171D24; z-index:2; margin-top:4px;"></div>
                {line_html}
              </div>
              <div style="flex:1; background:#202832; border:1px solid #2D3748; border-radius:6px; padding:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <span style="font-weight:bold; color:#FFFFFF; font-size:14px;">{title}</span>
                  <span style="font-size:11px; color:#A0AEC0;">{timestamp}</span>
                </div>
                <div style="font-size:12px; color:#C5A15A; margin:4px 0;">Agent/Model: {agent}</div>
                <div style="font-size:13px; color:#CBD5E0;">{details}</div>
                {f'<div style="margin-top:6px; font-family:monospace; font-size:11px; color:#718096;">SHA-256: {sha}</div>' if sha else ''}
              </div>
            </div>
            """)

        timeline_html = f"""
        <div style="background-color:#171D24; color:#E2E8F0; padding:16px; border-radius:8px; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
          <div style="font-size:16px; font-weight:bold; color:#C5A15A; border-bottom:1px solid #2D3748; padding-bottom:10px; margin-bottom:16px;">
            Epigraphical Provenance Chain: {html.escape(inscription_id)}
          </div>
          <div>
            {''.join(event_nodes_html)}
          </div>
        </div>
        """
        return timeline_html

