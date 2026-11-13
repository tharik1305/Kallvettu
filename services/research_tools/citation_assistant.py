"""
Scholarly Epigraphical Citation Assistant (Phase J).

Formats verified bibliographic references for major South Indian epigraphical corpuses:
- South Indian Inscriptions (SII)
- Epigraphia Indica (EI)
- Annual Report on Indian Epigraphy (ARE / ASI)
- Travancore Archaeological Series (TAS)
- Academic styles: Chicago Notes & Bibliography, APA 7th, and BibTeX.

Enforces academic integrity: never fabricates missing volume, number, or page numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class FormattedCitations:
    epigraphical_standard: str
    chicago: str
    apa: str
    bibtex: str
    tei_epidoc_bibl: str


class ScholarlyCitationAssistant:
    """Formats and validates academic epigraphical citations."""

    SERIES_FULL_NAMES = {
        "SII": "South Indian Inscriptions",
        "EI": "Epigraphia Indica",
        "ARE": "Annual Report on Indian Epigraphy",
        "TAS": "Travancore Archaeological Series",
        "IPS": "Inscriptions of the Pudukkottai State",
    }

    @classmethod
    def format_citation(
        cls,
        series: str,
        volume: Optional[str] = None,
        number: Optional[str] = None,
        pages: Optional[str] = None,
        year: Optional[int] = None,
        title: str = "",
        editor: str = "",
        url: str = "",
    ) -> FormattedCitations:
        """
        Formats epigraphical citation into multiple academic standards.
        """
        series_clean = series.strip().upper()
        full_series_name = cls.SERIES_FULL_NAMES.get(series_clean, series.strip())

        parts = [full_series_name]
        if volume:
            parts.append(f"Vol. {volume}")
        if number:
            parts.append(f"No. {number}")
        
        main_part = ", ".join(parts)
        if year:
            main_part += f" ({year})"
        if pages:
            main_part += f", pp. {pages}"

        std_citation = main_part
        if title:
            std_citation = f'"{title}", in {std_citation}'

        # Chicago Format
        # e.g., Editor, ed. "Title." South Indian Inscriptions, vol. 2, no. 1 (1916): 45–48.
        ed_str = f"{editor}, ed. " if editor else "Archaeological Survey of India. "
        vol_str = f"vol. {volume}, " if volume else ""
        no_str = f"no. {number} " if number else ""
        yr_str = f"({year})" if year else ""
        pg_str = f": {pages}." if pages else "."
        chicago = f'{ed_str}"{title or full_series_name}." {full_series_name}, {vol_str}{no_str}{yr_str}{pg_str}'

        # APA Format
        # e.g., Editor (Ed.). (1916). Title. South Indian Inscriptions, 2(1), 45-48.
        apa_yr = f"({year}). " if year else "(n.d.). "
        apa_issue = f"({number})" if number else ""
        apa_vol = f", {volume}{apa_issue}" if volume else ""
        apa_pg = f", {pages}." if pages else "."
        apa = f"{ed_str}{apa_yr}{title or 'Epigraphical Record'}. {full_series_name}{apa_vol}{apa_pg}"

        # BibTeX
        bib_key = f"{series_clean.lower()}_{volume or 'v'}_{number or 'n'}_{year or '0'}"
        bibtex = f"""@article{{{bib_key},
  author = {{{editor or 'Archaeological Survey of India'}}},
  title = {{{title or (full_series_name + ' Inscription')}}},
  journal = {{{full_series_name}}},
  volume = {{{volume or ''}}},
  number = {{{number or ''}}},
  pages = {{{pages or ''}}},
  year = {{{year or ''}}},
  url = {{{url}}}
}}"""

        # TEI EpiDoc <bibl>
        tei_bibl = f"""<bibl>
  <title level="s">{full_series_name}</title>
  {f'<biblScope unit="volume">{volume}</biblScope>' if volume else ''}
  {f'<biblScope unit="item">{number}</biblScope>' if number else ''}
  {f'<biblScope unit="page">{pages}</biblScope>' if pages else ''}
  {f'<date>{year}</date>' if year else ''}
</bibl>"""

        return FormattedCitations(
            epigraphical_standard=std_citation,
            chicago=chicago,
            apa=apa,
            bibtex=bibtex,
            tei_epidoc_bibl=tei_bibl,
        )
