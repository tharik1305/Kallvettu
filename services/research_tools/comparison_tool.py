"""
Epigraphical Comparison and Alignment Tool (Phase J).

Provides fine-grained side-by-side analysis between:
- Multi-engine OCR readings vs Epigraphist ground-truth readings.
- Diplomatic tier vs Normalized orthography.
- Distinct historical inscriptions or textual recensions.
"""

from __future__ import annotations

import difflib
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class ComparisonResult:
    text_a: str
    text_b: str
    similarity_ratio: float
    character_levenshtein: int
    jaccard_word_similarity: float
    diff_operations: List[Dict[str, Any]]
    side_by_side_html: str


class EpigraphicalComparisonTool:
    """Scholarly comparison and visual text alignment utility."""

    @classmethod
    def compare_readings(
        cls,
        text_a: str,
        text_b: str,
        label_a: str = "Reading A (Reference)",
        label_b: str = "Reading B (Hypothesis)",
    ) -> ComparisonResult:
        """
        Executes fine-grained comparative alignment between two epigraphical texts.
        """
        norm_a = unicodedata.normalize("NFC", text_a.strip())
        norm_b = unicodedata.normalize("NFC", text_b.strip())

        # Sequence matcher for char ratio
        sm = difflib.SequenceMatcher(None, norm_a, norm_b)
        ratio = round(sm.ratio(), 4)

        # Word Jaccard
        words_a = set(norm_a.split())
        words_b = set(norm_b.split())
        union = words_a.union(words_b)
        jaccard = round(len(words_a.intersection(words_b)) / float(max(1, len(union))), 4)

        # Edit distance
        diff_ops = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            diff_ops.append({
                "op": tag,
                "a_chunk": norm_a[i1:i2],
                "b_chunk": norm_b[j1:j2],
                "range_a": (i1, i2),
                "range_b": (j1, j2),
            })

        # Generate HTML side-by-side comparison
        html = cls._build_side_by_side_html(diff_ops, label_a, label_b)

        # Compute rough character distance
        char_dist = sum(
            max(len(op["a_chunk"]), len(op["b_chunk"]))
            for op in diff_ops
            if op["op"] != "equal"
        )

        return ComparisonResult(
            text_a=norm_a,
            text_b=norm_b,
            similarity_ratio=ratio,
            character_levenshtein=char_dist,
            jaccard_word_similarity=jaccard,
            diff_operations=diff_ops,
            side_by_side_html=html,
        )

    @classmethod
    def _build_side_by_side_html(
        cls,
        diff_ops: List[Dict[str, Any]],
        label_a: str,
        label_b: str,
    ) -> str:
        """Constructs styled HTML side-by-side comparison."""
        rendered_a = []
        rendered_b = []

        for op in diff_ops:
            tag = op["op"]
            a = op["a_chunk"]
            b = op["b_chunk"]

            if tag == "equal":
                rendered_a.append(f"<span>{a}</span>")
                rendered_b.append(f"<span>{b}</span>")
            elif tag == "replace":
                rendered_a.append(f'<span style="background-color:rgba(239, 83, 80, 0.3); border-bottom:2px solid #EF5350;" title="Replaced">{a}</span>')
                rendered_b.append(f'<span style="background-color:rgba(76, 175, 80, 0.3); border-bottom:2px solid #4CAF50;" title="Substituted with">{b}</span>')
            elif tag == "delete":
                rendered_a.append(f'<span style="background-color:rgba(239, 83, 80, 0.4); text-decoration:line-through;" title="Deleted">{a}</span>')
            elif tag == "insert":
                rendered_b.append(f'<span style="background-color:rgba(76, 175, 80, 0.4); border-bottom:2px solid #4CAF50;" title="Inserted">{b}</span>')

        html = f"""
        <div style="display:flex; gap:16px; font-family:monospace; font-size:14px; background:#171D24; padding:12px; border-radius:6px; color:#E2E8F0;">
          <div style="flex:1; border-right:1px solid #2D3748; padding-right:12px;">
            <div style="font-weight:bold; color:#C5A15A; margin-bottom:8px;">{label_a}</div>
            <div style="line-height:1.6; word-break:break-word;">{''.join(rendered_a)}</div>
          </div>
          <div style="flex:1; padding-left:4px;">
            <div style="font-weight:bold; color:#C5A15A; margin-bottom:8px;">{label_b}</div>
            <div style="line-height:1.6; word-break:break-word;">{''.join(rendered_b)}</div>
          </div>
        </div>
        """
        return html

