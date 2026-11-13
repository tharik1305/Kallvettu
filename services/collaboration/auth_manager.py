"""
Authentication & Role-Based Access Control Governance for Epigraphy Collaboration (Phase F).

Roles:
- RESEARCHER: Can upload inscriptions, perform restorations, submit transcriptions, add glyph annotations.
- REVIEWER: Can review, accept/reject transcriptions, adjudicate disputed K-IEFE slots, and approve annotations.
- ADMINISTRATOR: Can manage researcher profiles, freeze dataset versions, and manage system configurations.
"""

from __future__ import annotations

import enum
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, field

from services.database.models import ResearcherRole


@dataclass
class UserSession:
    """Active user session object."""
    user_id: str
    username: str
    full_name: str
    role: ResearcherRole
    institution: str = ""
    is_authenticated: bool = True

    def has_permission(self, permission: str) -> bool:
        """Evaluates whether the user's role grants the requested permission."""
        perms = ROLE_PERMISSIONS.get(self.role, set())
        return permission in perms


ROLE_PERMISSIONS: Dict[ResearcherRole, Set[str]] = {
    ResearcherRole.RESEARCHER: {
        "transcribe:create",
        "transcribe:edit_own",
        "annotate:create",
        "annotate:edit_own",
        "comment:create",
        "view:all",
        "export:dossier",
    },
    ResearcherRole.REVIEWER: {
        "transcribe:create",
        "transcribe:edit_own",
        "annotate:create",
        "annotate:edit_own",
        "comment:create",
        "view:all",
        "export:dossier",
        "review:adjudicate",
        "review:approve",
        "review:reject",
        "review:reassign",
    },
    ResearcherRole.ADMINISTRATOR: {
        "transcribe:create",
        "transcribe:edit_own",
        "annotate:create",
        "annotate:edit_own",
        "comment:create",
        "view:all",
        "export:dossier",
        "review:adjudicate",
        "review:approve",
        "review:reject",
        "review:reassign",
        "admin:manage_users",
        "admin:freeze_dataset",
        "admin:delete_records",
    },
}


DEFAULT_SCHOLAR_PROFILES = [
    {
        "id": "USR-SCHOLAR-01",
        "username": "epigraphist_raman",
        "full_name": "Dr. K. Raman (Epigraphist)",
        "role": ResearcherRole.REVIEWER,
        "institution": "Tamil Nadu State Department of Archaeology",
    },
    {
        "id": "USR-SCHOLAR-02",
        "username": "researcher_anand",
        "full_name": "Anand S. (Palaeographer)",
        "role": ResearcherRole.RESEARCHER,
        "institution": "International Institute of Tamil Studies",
    },
    {
        "id": "USR-ADMIN-01",
        "username": "kallvettu_admin",
        "full_name": "Principal Epigraphist / Administrator",
        "role": ResearcherRole.ADMINISTRATOR,
        "institution": "Kallvettu Research Consortium",
    },
]


def get_default_session(role: ResearcherRole = ResearcherRole.REVIEWER) -> UserSession:
    """Returns a default session for local standalone development."""
    for p in DEFAULT_SCHOLAR_PROFILES:
        if p["role"] == role:
            return UserSession(
                user_id=p["id"],
                username=p["username"],
                full_name=p["full_name"],
                role=p["role"],
                institution=p["institution"],
                is_authenticated=True,
            )
    return UserSession(
        user_id="USR-DEFAULT",
        username="epigraphist_guest",
        full_name="Guest Epigraphist",
        role=ResearcherRole.RESEARCHER,
        is_authenticated=True,
    )

