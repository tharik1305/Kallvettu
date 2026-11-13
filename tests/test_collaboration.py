"""
Tests for Collaborative Peer Review and Role-Based Access Governance (Phase F).
"""

import pytest
from services.database.connection import get_db_session
from services.database.models import (
    Inscription,
    Transcription,
    ResearcherRole,
    ReviewAction,
)
from services.collaboration.auth_manager import (
    UserSession,
    ROLE_PERMISSIONS,
    get_default_session,
)
from services.collaboration.review_service import ReviewService
from services.database.repository import EpigraphicalRepository


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_collab.db")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{test_db}")
    import services.database.connection as conn_mod
    conn_mod._ENGINE = None
    conn_mod._SESSION_FACTORY = None
    EpigraphicalRepository.init_database()
    yield
    conn_mod._ENGINE = None
    conn_mod._SESSION_FACTORY = None


def test_rbac_permissions():
    """Verify role permissions for RESEARCHER, REVIEWER, and ADMINISTRATOR."""
    researcher_session = UserSession(
        user_id="U1",
        username="scholar1",
        full_name="Scholar One",
        role=ResearcherRole.RESEARCHER,
    )
    reviewer_session = UserSession(
        user_id="U2",
        username="reviewer1",
        full_name="Reviewer One",
        role=ResearcherRole.REVIEWER,
    )
    admin_session = UserSession(
        user_id="U3",
        username="admin1",
        full_name="Admin One",
        role=ResearcherRole.ADMINISTRATOR,
    )

    # Researcher should not be able to adjudicate or manage users
    assert researcher_session.has_permission("transcribe:create") is True
    assert researcher_session.has_permission("review:adjudicate") is False
    assert researcher_session.has_permission("admin:manage_users") is False

    # Reviewer should be able to adjudicate but not manage users
    assert reviewer_session.has_permission("review:adjudicate") is True
    assert reviewer_session.has_permission("admin:manage_users") is False

    # Admin should have all permissions
    assert admin_session.has_permission("admin:manage_users") is True
    assert admin_session.has_permission("review:adjudicate") is True


def test_text_diff_computation():
    """Verify sequence matching and diff calculation between two readings."""
    text_a = "ஸ்வஸ்திஸ்ரீ கோப்பரகேஸரிபன்மற்க்கு"
    text_b = "ஸ்வஸ்திஸ்ரீ கோப்பரகேசரிபன்மர்க்கு"

    diff_result = ReviewService.compute_text_diff(text_a, text_b)
    assert diff_result["is_identical"] is False
    assert diff_result["similarity_ratio"] > 0.8
    assert len(diff_result["chunks"]) > 0

    identical_diff = ReviewService.compute_text_diff(text_a, text_a)
    assert identical_diff["is_identical"] is True
    assert identical_diff["similarity_ratio"] == 1.0


def test_adjudication_and_audit_flow():
    """Verify end-to-end adjudication, transcription update, revision logging, and audit trail."""
    test_insc_id = "TEST-INSC-COLLAB-01"
    test_tr_id = "TEST-TR-COLLAB-01"

    with get_db_session() as session:
        # Create test inscription and transcription
        insc = Inscription(
            id=test_insc_id,
            title_ta="சோழர் காலக் கல்வெட்டு",
            title_en="Collab Test Inscription",
            script_category="Chola Tamil",
            site_name="Thanjavur Brihadisvara",
            district="Thanjavur",
        )
        session.merge(insc)

        tr = Transcription(
            id=test_tr_id,
            inscription_id=test_insc_id,
            text_content="ஸ்வஸ்திஸ்ரீ திரிபுவன சக்கரவர்த்திகள்",
            provenance="Automated AI Pipeline",
            is_primary=True,
        )
        session.merge(tr)
        session.commit()

    # Perform adjudication
    rev = ReviewService.submit_adjudication(
        transcription_id=test_tr_id,
        reviewer_id="USR-SCHOLAR-01",
        action=ReviewAction.EDIT_SCHOLAR,
        approved_reading="ஸ்வஸ்திஸ்ரீ திரிபுவன சக்ரவர்த்திகள் ஸ்ரீ ராஜராஜ தேவர்க்கு",
        previous_reading="ஸ்வஸ்திஸ்ரீ திரிபுவன சக்கரவர்த்திகள்",
        comments="Corrected ligature and completed missing royal epigraphical formula.",
    )

    assert rev.id.startswith("REV-")
    assert rev.action == ReviewAction.EDIT_SCHOLAR

    # Verify transcription text updated
    with get_db_session() as session:
        updated_tr = session.query(Transcription).filter(Transcription.id == test_tr_id).first()
        assert updated_tr is not None
        assert "ராஜராஜ" in updated_tr.text_content
        assert "Scholar Adjudicated" in updated_tr.provenance

    # Verify revision history
    history = ReviewService.get_revision_history(test_tr_id)
    assert len(history) >= 1
    assert history[0]["changed_by"] == "USR-SCHOLAR-01"
    assert "ராஜராஜ" in history[0]["new_state"]["text_content"]

    # Verify audit event
    audits = ReviewService.list_audit_events_for_entity(test_tr_id)
    assert len(audits) >= 1
    assert audits[0]["user_id"] == "USR-SCHOLAR-01"
    assert len(audits[0]["sha256_signature"]) == 64
