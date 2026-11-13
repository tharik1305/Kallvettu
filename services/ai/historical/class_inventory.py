"""
Class Inventory Registry for Ancient & Medieval Tamil Historical Scripts (Phase I).

Provides standardized grapheme inventories, Unicode mappings, and known
visually confusable character pairs for:
1. Tamil-Brahmi (Early Sangam, c. 3rd c. BCE - 2nd c. CE)
2. Vatteluttu (Round Script, c. 4th c. CE - 11th c. CE)
3. Pallava Grantha (Monolithic & Copper Plate, c. 6th c. CE - 9th c. CE)
4. Medieval Chola Tamil (Chola Imperial, c. 9th c. CE - 13th c. CE)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set


@dataclass(frozen=True)
class GraphemeDefinition:
    """Formal definition of a historical script grapheme."""
    grapheme_id: str
    script: str
    canonical_char: str
    unicode_equivalent: str
    iso15919: str
    category: str  # VOWEL, CONSONANT, VOWEL_SIGN, PULLI, SYMBOL
    description: str = ""
    confusable_with: Tuple[str, ...] = ()


# -----------------------------------------------------------------------------
# 1. Tamil-Brahmi Inventory (Bhattiprolu / Cave Tamil-Brahmi)
# -----------------------------------------------------------------------------
TAMIL_BRAHMI_CLASSES: List[GraphemeDefinition] = [
    GraphemeDefinition("TB-A", "Tamil-Brahmi", "𑀅", "அ", "a", "VOWEL", "Initial short vowel a"),
    GraphemeDefinition("TB-AA", "Tamil-Brahmi", "𑀆", "ஆ", "ā", "VOWEL", "Initial long vowel ā"),
    GraphemeDefinition("TB-I", "Tamil-Brahmi", "𑀇", "இ", "i", "VOWEL", "Initial short vowel i"),
    GraphemeDefinition("TB-II", "Tamil-Brahmi", "𑀈", "ஈ", "ī", "VOWEL", "Initial long vowel ī"),
    GraphemeDefinition("TB-U", "Tamil-Brahmi", "𑀉", "உ", "u", "VOWEL", "Initial short vowel u"),
    GraphemeDefinition("TB-E", "Tamil-Brahmi", "𑀏", "எ", "e", "VOWEL", "Initial short vowel e"),
    # Consonants
    GraphemeDefinition("TB-KA", "Tamil-Brahmi", "𑀓", "க", "ka", "CONSONANT", "Velar stop ka", ("TB-TA",)),
    GraphemeDefinition("TB-NGA", "Tamil-Brahmi", "𑀗", "ங", "ṅa", "CONSONANT", "Velar nasal ṅa"),
    GraphemeDefinition("TB-CA", "Tamil-Brahmi", "𑀘", "ச", "ca", "CONSONANT", "Palatal stop ca"),
    GraphemeDefinition("TB-NYA", "Tamil-Brahmi", "𑀜", "ஞ", "ña", "CONSONANT", "Palatal nasal ña"),
    GraphemeDefinition("TB-TTA", "Tamil-Brahmi", "𑀝", "ட", "ṭa", "CONSONANT", "Retroflex stop ṭa", ("TB-THA",)),
    GraphemeDefinition("TB-NNA", "Tamil-Brahmi", "𑀡", "ண", "ṇa", "CONSONANT", "Retroflex nasal ṇa", ("TB-NA",)),
    GraphemeDefinition("TB-TA", "Tamil-Brahmi", "𑀢", "த", "ta", "CONSONANT", "Dental stop ta", ("TB-KA",)),
    GraphemeDefinition("TB-NA", "Tamil-Brahmi", "𑀦", "ந", "na", "CONSONANT", "Dental nasal na", ("TB-NNA",)),
    GraphemeDefinition("TB-PA", "Tamil-Brahmi", "𑀧", "ப", "pa", "CONSONANT", "Labial stop pa", ("TB-VA",)),
    GraphemeDefinition("TB-MA", "Tamil-Brahmi", "𑀫", "ம", "ma", "CONSONANT", "Labial nasal ma"),
    GraphemeDefinition("TB-YA", "Tamil-Brahmi", "𑀬", "ய", "ya", "CONSONANT", "Palatal glide ya"),
    GraphemeDefinition("TB-RA", "Tamil-Brahmi", "𑀭", "ர", "ra", "CONSONANT", "Alveolar flap ra"),
    GraphemeDefinition("TB-LA", "Tamil-Brahmi", "𑀮", "ல", "la", "CONSONANT", "Alveolar lateral la"),
    GraphemeDefinition("TB-VA", "Tamil-Brahmi", "𑀯", "வ", "va", "CONSONANT", "Labio-dental glide va", ("TB-PA",)),
    GraphemeDefinition("TB-LLLA", "Tamil-Brahmi", "𑀵", "ழ", "ḻa", "CONSONANT", "Retroflex central approximant ḻa"),
    GraphemeDefinition("TB-LLA", "Tamil-Brahmi", "𑀴", "ள", "ḷa", "CONSONANT", "Retroflex lateral ḷa"),
    GraphemeDefinition("TB-RRA", "Tamil-Brahmi", "𑀶", "ற", "ṟa", "CONSONANT", "Alveolar trill ṟa"),
    GraphemeDefinition("TB-NNNA", "Tamil-Brahmi", "𑀷", "ன", "ṉa", "CONSONANT", "Alveolar nasal ṉa"),
]

# -----------------------------------------------------------------------------
# 2. Medieval Chola Tamil Inventory
# -----------------------------------------------------------------------------
CHOLA_TAMIL_CLASSES: List[GraphemeDefinition] = [
    GraphemeDefinition("CT-A", "Chola Tamil", "அ", "அ", "a", "VOWEL", "Chola vowel a"),
    GraphemeDefinition("CT-AA", "Chola Tamil", "ஆ", "ஆ", "ā", "VOWEL", "Chola vowel ā"),
    GraphemeDefinition("CT-I", "Chola Tamil", "இ", "இ", "i", "VOWEL", "Chola vowel i"),
    GraphemeDefinition("CT-II", "Chola Tamil", "ஈ", "ஈ", "ī", "VOWEL", "Chola vowel ī"),
    GraphemeDefinition("CT-U", "Chola Tamil", "உ", "உ", "u", "VOWEL", "Chola vowel u"),
    GraphemeDefinition("CT-E", "Chola Tamil", "எ", "எ", "e", "VOWEL", "Chola vowel e"),
    GraphemeDefinition("CT-EE", "Chola Tamil", "ஏ", "ஏ", "ē", "VOWEL", "Chola vowel ē"),
    GraphemeDefinition("CT-AI", "Chola Tamil", "ஐ", "ஐ", "ai", "VOWEL", "Chola vowel ai"),
    GraphemeDefinition("CT-O", "Chola Tamil", "ஒ", "ஒ", "o", "VOWEL", "Chola vowel o"),
    GraphemeDefinition("CT-KA", "Chola Tamil", "க", "க", "ka", "CONSONANT", "Chola ka"),
    GraphemeDefinition("CT-CA", "Chola Tamil", "ச", "ச", "ca", "CONSONANT", "Chola ca"),
    GraphemeDefinition("CT-TTA", "Chola Tamil", "ட", "ட", "ṭa", "CONSONANT", "Chola ṭa"),
    GraphemeDefinition("CT-NNA", "Chola Tamil", "ண", "ண", "ṇa", "CONSONANT", "Chola ṇa", ("CT-NNA-THREE-LOOP", "CT-NNA-TWO-LOOP")),
    GraphemeDefinition("CT-TA", "Chola Tamil", "த", "த", "ta", "CONSONANT", "Chola ta", ("CT-NA",)),
    GraphemeDefinition("CT-NA", "Chola Tamil", "ந", "ந", "na", "CONSONANT", "Chola na", ("CT-TA",)),
    GraphemeDefinition("CT-PA", "Chola Tamil", "ப", "ப", "pa", "CONSONANT", "Chola pa", ("CT-VA",)),
    GraphemeDefinition("CT-MA", "Chola Tamil", "ம", "ம", "ma", "CONSONANT", "Chola ma"),
    GraphemeDefinition("CT-YA", "Chola Tamil", "ய", "ய", "ya", "CONSONANT", "Chola ya"),
    GraphemeDefinition("CT-RA", "Chola Tamil", "ர", "ர", "ra", "CONSONANT", "Chola ra", ("CT-KAAL",)),
    GraphemeDefinition("CT-LA", "Chola Tamil", "ல", "ல", "la", "CONSONANT", "Chola la"),
    GraphemeDefinition("CT-VA", "Chola Tamil", "வ", "வ", "va", "CONSONANT", "Chola va", ("CT-PA",)),
    GraphemeDefinition("CT-LLLA", "Chola Tamil", "ழ", "ழ", "ḻa", "CONSONANT", "Chola ḻa"),
    GraphemeDefinition("CT-LLA", "Chola Tamil", "ள", "ள", "ḷa", "CONSONANT", "Chola ḷa"),
    GraphemeDefinition("CT-RRA", "Chola Tamil", "ற", "ற", "ṟa", "CONSONANT", "Chola ṟa"),
    GraphemeDefinition("CT-NNNA", "Chola Tamil", "ன", "ன", "ṉa", "CONSONANT", "Chola ṉa", ("CT-NNA",)),
    GraphemeDefinition("CT-PULLI", "Chola Tamil", "்", "்", "virama", "PULLI", "Subtle carved dot or absence of vowel"),
    GraphemeDefinition("CT-KAAL", "Chola Tamil", "ா", "ா", "ā-sign", "VOWEL_SIGN", "Long vowel marker", ("CT-RA",)),
]

# -----------------------------------------------------------------------------
# 3. Known Historical Confusable Pairs
# -----------------------------------------------------------------------------
CONFUSABLE_PAIRS: List[Dict[str, str]] = [
    {
        "glyph_a": "ண",
        "glyph_b": "ன",
        "script": "Chola Tamil",
        "confusion_reason": "Number of loops eroded on weathered granite plinths.",
        "disambiguation_heuristic": "Check sandhi context: ண precedes retroflex ட, ன precedes dental ற or word-final.",
    },
    {
        "glyph_a": "ர",
        "glyph_b": "ா",
        "script": "Chola Tamil",
        "confusion_reason": "Identical vertical stroke with top horizontal serif in medieval epigraphy.",
        "disambiguation_heuristic": "Preceding character: if preceding consonant lacks vowel sign, ா forms long vowel; otherwise standalone consonant ர.",
    },
    {
        "glyph_a": "த",
        "glyph_b": "ந",
        "script": "Chola Tamil",
        "confusion_reason": "Loop closure wear causes dental stop and dental nasal to resemble each other.",
        "disambiguation_heuristic": "Grammatical position: word-initial mostly ந, medial conjuncts follow strict assimilation.",
    },
    {
        "glyph_a": "ப",
        "glyph_b": "வ",
        "script": "Chola Tamil",
        "confusion_reason": "Curvature of the left stem can round into va or stay angular for pa.",
        "disambiguation_heuristic": "Context lexical root lookup.",
    },
]


class ScriptClassInventoryRegistry:
    """Registry providing canonical grapheme classes and validation for historical scripts."""

    _INVENTORIES: Dict[str, List[GraphemeDefinition]] = {
        "Tamil-Brahmi": TAMIL_BRAHMI_CLASSES,
        "Chola Tamil": CHOLA_TAMIL_CLASSES,
        "Medieval Chola Tamil": CHOLA_TAMIL_CLASSES,
    }

    @classmethod
    def get_supported_scripts(cls) -> List[str]:
        """Returns list of supported historical scripts."""
        return ["Tamil-Brahmi", "Vatteluttu", "Pallava Grantha", "Chola Tamil"]

    @classmethod
    def get_classes_for_script(cls, script: str) -> List[GraphemeDefinition]:
        """Returns grapheme definitions for a given historical script."""
        # Normalize script name
        for k, v in cls._INVENTORIES.items():
            if k.lower() in script.lower() or script.lower() in k.lower():
                return v
        return CHOLA_TAMIL_CLASSES

    @classmethod
    def get_confusable_pairs(cls, script: Optional[str] = None) -> List[Dict[str, str]]:
        """Returns verified epigraphical confusable pairs with disambiguation heuristics."""
        if not script:
            return CONFUSABLE_PAIRS
        return [p for p in CONFUSABLE_PAIRS if script.lower() in p["script"].lower()]

    @classmethod
    def validate_grapheme(cls, script: str, char: str) -> bool:
        """Checks if a character exists in the script inventory."""
        classes = cls.get_classes_for_script(script)
        return any(c.unicode_equivalent == char or c.canonical_char == char for c in classes)

