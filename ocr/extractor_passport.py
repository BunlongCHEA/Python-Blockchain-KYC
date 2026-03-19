"""
Field extraction for passports using MRZ (Machine Readable Zone)
and fallback text-search patterns.

MRZ format reference (ICAO Doc 9303):
  Line 1: P<KHM<LASTNAME<<FIRSTNAME<<<<<<<<<<<<<<<
  Line 2: PASSNO<<<<0KHM9001011M2501010<<<<<<<<<<<<4
"""
import re
from typing import Any, Dict, List


# ── MRZ helpers ────────────────────────────────────────────────────────────────

_MRZ1_PATTERN = re.compile(r"P[<A-Z][A-Z<]{38}")
_MRZ2_PATTERN = re.compile(
    r"([A-Z0-9<]{9})"   # passport number (9 chars)
    r"(\d)"             # check digit
    r"([A-Z]{3})"       # nationality
    r"(\d{6})"          # date of birth YYMMDD
    r"(\d)"             # check digit
    r"([MF<])"          # sex
    r"(\d{6})"          # expiry YYMMDD
)


def _parse_mrz_name(raw: str) -> tuple[str, str]:
    """Split a MRZ name segment into (last_name, first_name)."""
    parts = raw.split("<<", 1)
    last  = parts[0].replace("<", " ").strip() if len(parts) > 0 else ""
    first = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
    return last, first


def _yymmdd_to_date(s: str) -> str:
    """Convert YYMMDD → DD/MM/YYYY (century assumed ≥ 1924)."""
    if len(s) != 6:
        return s
    yy, mm, dd = s[:2], s[2:4], s[4:]
    century = "20" if int(yy) <= 24 else "19"
    return f"{dd}/{mm}/{century}{yy}"


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_passport_fields(texts: List[str]) -> Dict[str, Any]:
    """
    Parse OCR text list from a passport image into structured fields.

    Priority:
    1. MRZ lines (most reliable)
    2. Text-based fallback patterns
    """
    full_joined  = "\n".join(texts)
    full_no_space = full_joined.upper().replace(" ", "")
    fields: Dict[str, Any] = {}

    # ── 1. MRZ Line 1 — name ──────────────────────────────
    m1 = _MRZ1_PATTERN.search(full_no_space)
    if m1:
        raw_name = m1.group(0)[5:]   # strip "P<KHM" or similar header
        last, first = _parse_mrz_name(raw_name)
        if last:
            fields["last_name"]  = last
        if first:
            fields["first_name"] = first

    # ── 2. MRZ Line 2 — number / DOB / expiry ────────────
    m2 = _MRZ2_PATTERN.search(full_no_space)
    if m2:
        g = m2.groups()
        fields["passport_number"] = g[0].replace("<", "")
        fields["nationality"]     = g[2]
        fields["date_of_birth"]   = _yymmdd_to_date(g[3])
        fields["sex"]             = g[5] if g[5] != "<" else ""
        fields["expiry_date"]     = _yymmdd_to_date(g[6])

    # ── 3. Text-based fallbacks ───────────────────────────
    if not fields.get("last_name"):
        m = re.search(
            r"(?:Surname|Family\s*Name)\s*[:\-]?\s*([A-Za-z\s\-]{2,40})",
            full_joined, re.IGNORECASE,
        )
        if m:
            fields["last_name"] = m.group(1).strip()

    if not fields.get("first_name"):
        m = re.search(
            r"(?:Given\s*Names?|First\s*Name)\s*[:\-]?\s*([A-Za-z\s\-]{2,40})",
            full_joined, re.IGNORECASE,
        )
        if m:
            fields["first_name"] = m.group(1).strip()

    if not fields.get("date_of_birth"):
        m = re.search(
            r"(?:Date\s*of\s*Birth|DOB)\s*[:\-]?\s*"
            r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
            full_joined, re.IGNORECASE,
        )
        if m:
            fields["date_of_birth"] = m.group(1).strip()

    if not fields.get("passport_number"):
        m = re.search(r"\b([A-Z]{1,2}\d{7,9})\b", full_joined.upper())
        if m:
            fields["passport_number"] = m.group(1)

    return fields