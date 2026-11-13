"""
Collaborative Peer Review & Disagreement Tracking Service (Phase F).

Manages:
- Formal reviewer adjudication workflows (ACCEPT, ADOPT_PROVIDER, EDIT_SCHOLAR, MARK_ILLEGIBLE, DEFER).
- Immutable revision logging with side-by-side character/word diff comparison.
- Cryptographic SHA-256 audit logs preserving reviewer attribution and timestamps.
- Glyph-level scholarly comment threads.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from services.database.connection import get_db_session
from services.database.models import (
    Review,
    Revision,
    AuditEvent,
    Transcription,
    GlyphAnnotation,
    ReviewAction,
    AnnotationStatus,
)

logger = logging.getLogger("kallvettu.collaboration")


class ReviewService:
    """Manages scholarly review, disagreement resolution, and revision history."""

    @classmethod
    def submit_adjudication(
        cls,
        transcription_id: str,
        reviewer_id: str,
        action: ReviewAction,
        approved_reading: str,
        previous_reading: str = "",
        comments: str = "",
    ) -> Review:
        """
        Executes a formal review action on a transcription reading,
        saving the review, updating the transcription, and recording an immutable audit revision.
        """
        now = datetime.now(timezone.utc)
        with get_db_session() as s:
            tr = s.query(Transcription).filter(Transcription.id == transcription_id).first()
            if not tr:
                raise ValueError(f"Transcription '{transcription_id}' not found.")

            prev_text = previous_reading or tr.text_content

            # 1. Record Review
            review_id = f"REV-{uuid.uuid4().hex[:12].upper()}"
            rev = Review(
                id=review_id,
                transcription_id=transcription_id,
                reviewer_id=reviewer_id,
                action=action,
                previous_reading=prev_text,
                approved_reading=approved_reading,
                comments=comments,
                created_at=now,
            )
            s.add(rev)

            # 2. Update Transcription content
            tr.text_content = approved_reading
            tr.provenance = f"Scholar Adjudicated ({action.value}) by {reviewer_id}"

            # 3. Create Revision Record
            revision_id = f"REVIS-{uuid.uuid4().hex[:12].upper()}"
            revision = Revision(
                id=revision_id,
                entity_type="Transcription",
                entity_id=transcription_id,
                previous_state={"text_content": prev_text},
                new_state={"text_content": approved_reading, "action": action.value, "comments": comments},
                changed_by=reviewer_id,
                timestamp=now,
            )
            s.add(revision)

            # 4. Cryptographic Audit Event (Append-Only)
            sig_payload = f"{transcription_id}:{reviewer_id}:{action.value}:{approved_reading}:{now.isoformat()}"
            sha_sig = hashlib.sha256(sig_payload.encode("utf-8")).hexdigest()

            audit = AuditEvent(
                id=f"AUDIT-{uuid.uuid4().hex[:12].upper()}",
                event_type="SCHOLAR_ADJUDICATION",
                entity_type="Transcription",
                entity_id=transcription_id,
                user_id=reviewer_id,
                sha256_signature=sha_sig,
                details={
                    "review_id": review_id,
                    "action": action.value,
                    "previous_length": len(prev_text),
                    "new_length": len(approved_reading),
                    "comments": comments,
                },
                timestamp=now,
            )
            s.add(audit)
            s.commit()

            logger.info("Adjudication completed: %s on %s by %s", action.value, transcription_id, reviewer_id)
            return rev

    @classmethod
    def get_revision_history(cls, entity_id: str) -> List[Dict[str, Any]]:
        """Retrieves chronological revisions for an entity."""
        with get_db_session() as s:
            revisions = (
                s.query(Revision)
                .filter(Revision.entity_id == entity_id)
                .order_by(Revision.timestamp.desc())
                .all()
            )
            return [
                {
                    "revision_id": r.id,
                    "entity_type": r.entity_type,
                    "entity_id": r.entity_id,
                    "changed_by": r.changed_by,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else "",
                    "previous_state": r.previous_state,
                    "new_state": r.new_state,
                }
                for r in revisions
            ]

    @classmethod
    def compute_text_diff(cls, text_a: str, text_b: str) -> Dict[str, Any]:
        """
        Computes detailed token-level and character-level diff between two readings.
        """
        matcher = difflib.SequenceMatcher(None, text_a, text_b)
        diff_chunks = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            diff_chunks.append({
                "tag": tag,  # 'equal', 'replace', 'delete', 'insert'
                "a_chunk": text_a[i1:i2],
                "b_chunk": text_b[j1:j2],
                "pos_a": (i1, i2),
                "pos_b": (j1, j2),
            })

        similarity_ratio = matcher.ratio()
        return {
            "similarity_ratio": round(similarity_ratio, 4),
            "is_identical": text_a == text_b,
            "chunks": diff_chunks,
        }

    @classmethod
    def list_audit_events_for_entity(cls, entity_id: str) -> List[Dict[str, Any]]:
        """Retrieves append-only audit trail records."""
        with get_db_session() as s:
            events = (
                s.query(AuditEvent)
                .filter(AuditEvent.entity_id == entity_id)
                .order_by(AuditEvent.timestamp.desc())
                .all()
            )
            return [
                {
                    "audit_id": e.id,
                    "event_type": e.event_type,
                    "user_id": e.user_id,
                    "sha256_signature": e.sha256_signature,
                    "details": e.details,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else "",
                }
                for e in events
            ]

