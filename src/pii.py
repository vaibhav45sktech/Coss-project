"""
Rule-based PII detection for Indian agricultural data.

Layer 1 of a three-layer PII pipeline:
- Layer 1 (this file): India-specific regex + gazetteer detection
- Layer 2 (later): Microsoft Presidio for English NLP-based entities
- Layer 3 (later): Audit sampling + per-session consistent placeholder mapping

Each detector returns PIIMatch objects with spans, original text, and confidence.
Detection is separated from redaction — this module only finds PII, the pipeline
class (added later) handles replacement strategy.
"""

import re
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PIIMatch:
    """A single PII detection result.

    `source` will be 'rule' for this module, 'presidio' once Layer 2 lands.
    Confidence is a 0-1 score letting the pipeline prioritize among overlapping
    detections from different sources.
    """
    pii_type: str
    start: int
    end: int
    original: str
    confidence: float
    source: str = "rule"


# ---------------------------------------------------------------------------
# Verhoeff checksum (used by Aadhaar)
# ---------------------------------------------------------------------------
# Aadhaar's last digit is a Verhoeff checksum over the first 11 digits.
# Real Aadhaar numbers pass this check; random 12-digit numbers usually don't.
# We use the result to distinguish "likely real Aadhaar" (conf 0.98) from
# "looks like Aadhaar but probably synthetic" (conf 0.55 — still worth redacting).

D_TABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

P_TABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_check(number_str: str) -> bool:
    """Return True if number_str passes the Verhoeff checksum."""
    if not number_str.isdigit():
        return False
    c = 0
    for i, digit in enumerate(reversed(number_str)):
        c = D_TABLE[c][P_TABLE[i % 8][int(digit)]]
    return c == 0


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class IndianPIIRules:
    """Rule-based PII detection layer tuned for Indian agricultural context.

    Loads name + village gazetteers once at construction. All regex patterns
    are compiled once for performance — this matters when we run across
    thousands of trajectories in the full pipeline.
    """

    # Pre-defined keyword sets used across detectors
    AMOUNT_KEYWORDS = ("rs", "rs.", "inr", "rupees", "₹")
    BANK_KEYWORDS = ("account", "a/c", "bank", "ifsc", "savings", "current")

    def __init__(self, gazetteer_dir: Path = Path("data/gazetteers")):
        self.names = self._load_gazetteer(gazetteer_dir / "indian_names.txt")
        self.villages = self._load_gazetteer(gazetteer_dir / "indian_villages.txt")
        self._compile_patterns()
        logger.info(
            f"IndianPIIRules ready: {len(self.names)} names, "
            f"{len(self.villages)} villages loaded"
        )

    # ---- gazetteer loading ----

    @staticmethod
    def _load_gazetteer(path: Path) -> set[str]:
        if not path.exists():
            logger.warning(f"Gazetteer file missing: {path} — returning empty set")
            return set()
        with path.open(encoding="utf-8") as fh:
            return {line.strip().lower() for line in fh if line.strip()}

    # ---- pattern compilation ----

    def _compile_patterns(self) -> None:
        # Phone: optional +91/91 prefix, optional separator, then 10 digits
        # starting with 6/7/8/9, optionally with a space or dash in the middle.
        self._phone_re = re.compile(
            r"(?:\+?91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}\b"
        )

        # Aadhaar: 12 digits, optionally grouped 4-4-4 with spaces or dashes.
        # We validate the digits via Verhoeff after stripping separators.
        self._aadhaar_re = re.compile(
            r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
        )

        # PAN: 5 letters, 4 digits, 1 letter
        self._pan_re = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")

        # Email
        self._email_re = re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
        )

        # GPS: two signed decimals separated by comma + optional space.
        # We validate the bounding box (India) after matching.
        self._gps_re = re.compile(
            r"(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)"
        )

        # Indian pincode: 6 digits, first digit 1-8, word-bounded
        self._pincode_re = re.compile(r"\b[1-8]\d{5}\b")

        # IFSC: 4 letters + 0 + 6 alphanumeric
        self._ifsc_re = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")

        # Word tokenizer used by gazetteer-based detectors
        self._word_re = re.compile(r"\b[A-Za-z][A-Za-z\-]*\b")

    # ---- individual detectors ----

    def detect_phone(self, text: str) -> list[PIIMatch]:
        results = []
        for m in self._phone_re.finditer(text):
            # Skip if preceded by an amount keyword within 10 chars
            preceding = text[max(0, m.start() - 10): m.start()].lower()
            if any(kw in preceding for kw in self.AMOUNT_KEYWORDS):
                continue
            confidence = 0.95 if m.group().lstrip().startswith(("+91", "91")) else 0.85
            results.append(PIIMatch(
                pii_type="PHONE",
                start=m.start(),
                end=m.end(),
                original=m.group(),
                confidence=confidence,
            ))
        return results

    def detect_aadhaar(self, text: str) -> list[PIIMatch]:
        results = []
        for m in self._aadhaar_re.finditer(text):
            digits = re.sub(r"[\s\-]", "", m.group())
            if len(digits) != 12:
                continue
            confidence = 0.98 if verhoeff_check(digits) else 0.55
            results.append(PIIMatch(
                pii_type="AADHAAR",
                start=m.start(),
                end=m.end(),
                original=m.group(),
                confidence=confidence,
            ))
        return results

    def detect_pan(self, text: str) -> list[PIIMatch]:
        return [
            PIIMatch("PAN", m.start(), m.end(), m.group(), 0.95)
            for m in self._pan_re.finditer(text)
        ]

    def detect_email(self, text: str) -> list[PIIMatch]:
        return [
            PIIMatch("EMAIL", m.start(), m.end(), m.group(), 0.95)
            for m in self._email_re.finditer(text)
        ]

    def detect_gps(self, text: str) -> list[PIIMatch]:
        results = []
        for m in self._gps_re.finditer(text):
            try:
                lat = float(m.group(1))
                lng = float(m.group(2))
            except ValueError:
                continue
            # India bounding box check
            if 6.0 <= lat <= 37.0 and 68.0 <= lng <= 97.0:
                results.append(PIIMatch(
                    pii_type="GPS",
                    start=m.start(),
                    end=m.end(),
                    original=m.group(),
                    confidence=0.90,
                ))
        return results

    def detect_pincode(self, text: str) -> list[PIIMatch]:
        return [
            PIIMatch("PINCODE", m.start(), m.end(), m.group(), 0.70)
            for m in self._pincode_re.finditer(text)
        ]

    def detect_ifsc(self, text: str) -> list[PIIMatch]:
        return [
            PIIMatch("IFSC", m.start(), m.end(), m.group(), 0.95)
            for m in self._ifsc_re.finditer(text)
        ]

    def detect_bank_account(self, text: str) -> list[PIIMatch]:
        """Match 9-18 digit sequences near banking keywords."""
        results = []
        lower = text.lower()
        for m in re.finditer(r"\b\d{9,18}\b", text):
            # Look ±30 chars for banking context
            window_start = max(0, m.start() - 30)
            window_end = min(len(text), m.end() + 30)
            window = lower[window_start:window_end]
            if any(kw in window for kw in self.BANK_KEYWORDS):
                results.append(PIIMatch(
                    pii_type="BANK_ACCOUNT",
                    start=m.start(),
                    end=m.end(),
                    original=m.group(),
                    confidence=0.70,
                ))
        return results

    def detect_names(self, text: str) -> list[PIIMatch]:
        if not self.names:
            return []
        results = []
        for m in self._word_re.finditer(text):
            if m.group().lower() in self.names:
                results.append(PIIMatch(
                    pii_type="NAME",
                    start=m.start(),
                    end=m.end(),
                    original=m.group(),
                    confidence=0.70,
                ))
        return results

    def detect_villages(self, text: str) -> list[PIIMatch]:
        if not self.villages:
            return []
        results = []
        for m in self._word_re.finditer(text):
            if m.group().lower() in self.villages:
                results.append(PIIMatch(
                    pii_type="LOCATION",
                    start=m.start(),
                    end=m.end(),
                    original=m.group(),
                    confidence=0.75,
                ))
        return results

    # ---- aggregator ----

    def detect_all(self, text: str) -> list[PIIMatch]:
        """Run every detector and return matches sorted by start position.

        Does NOT deduplicate overlapping matches — that's the pipeline's job
        when merging across Layer 1 (this) and Layer 2 (Presidio).
        """
        matches: list[PIIMatch] = []
        matches.extend(self.detect_phone(text))
        matches.extend(self.detect_aadhaar(text))
        matches.extend(self.detect_pan(text))
        matches.extend(self.detect_email(text))
        matches.extend(self.detect_gps(text))
        matches.extend(self.detect_pincode(text))
        matches.extend(self.detect_ifsc(text))
        matches.extend(self.detect_bank_account(text))
        matches.extend(self.detect_names(text))
        matches.extend(self.detect_villages(text))
        return sorted(matches, key=lambda m: m.start)


class PresidioPIIDetector:
    """Wraps Microsoft Presidio for NLP-based entity detection.

    Catches English PERSON / LOCATION / ORGANIZATION entities that our
    regex+gazetteer layer would miss (e.g. unfamiliar names, foreign
    locations, company names). Initialized lazily — Presidio's AnalyzerEngine
    takes 1-2 seconds to load the spaCy model, so we do it once per process.
    """

    # Map Presidio's entity labels to our internal PIIMatch types.
    # Keeping a tight allowlist avoids noisy matches (URL, DATE_TIME, etc).
    ENTITY_MAP = {
        "PERSON": "NAME",
        "LOCATION": "LOCATION",
        "ORGANIZATION": "ORGANIZATION",
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE",
        "CREDIT_CARD": "CREDIT_CARD",
    }

    # Words Presidio sometimes mis-tags as PERSON in Indian/agri context.
    # These are domain vocabulary, not real names — drop them.
    PERSON_STOPWORDS = {
        "aadhaar", "pan", "pm-kisan", "pm", "kisan", "msp",
        "kharif", "rabi", "zaid", "rtc", "khasra", "ifsc",
        "hdfc", "sbi", "icici", "axis",  # banks
        "mandi", "fasal", "kheti", "paani",
    }

    def __init__(self):
        from presidio_analyzer import AnalyzerEngine
        self._analyzer = AnalyzerEngine()
        logger.info("Presidio analyzer initialized")

    def detect(self, text: str) -> list[PIIMatch]:
        try:
            results = self._analyzer.analyze(
                text=text,
                language="en",
                entities=list(self.ENTITY_MAP.keys()),
            )
        except Exception as e:
            logger.warning(f"Presidio analysis failed on text snippet: {e}")
            return []

        matches = []
        for r in results:
            our_type = self.ENTITY_MAP.get(r.entity_type)
            if our_type is None:
                continue
            matched_text = text[r.start:r.end]
            if our_type == "NAME" and matched_text.lower().strip() in self.PERSON_STOPWORDS:
                continue
            matches.append(PIIMatch(
                pii_type=our_type,
                start=r.start,
                end=r.end,
                original=matched_text,
                confidence=float(r.score),
                source="presidio",
            ))
        return matches


# ===========================================================================
# Layer 3: PIIPipeline — orchestrator with dedup and per-session placeholders
# ===========================================================================

class PIIPipeline:
    """Runs both detection layers, dedupes overlapping spans, and redacts text.

    Placeholder mapping is per-pipeline-instance, so a fresh pipeline per
    session gives consistent within-session placeholders (Ramesh → <NAME_1>
    every time he's mentioned in that session) while keeping sessions
    isolated from each other.

    Usage:
        pipeline = PIIPipeline()                  # shares dedup across all redactions
        pipeline.reset_session()                  # clear placeholder map per session
        redacted, matches = pipeline.redact(text)
    """

    def __init__(self, use_presidio: bool = True):
        self.rules = IndianPIIRules()
        self.presidio = PresidioPIIDetector() if use_presidio else None
        self._placeholder_map: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {}

    def reset_session(self) -> None:
        """Call at the start of each new session to reset placeholder IDs."""
        self._placeholder_map = {}
        self._counters = {}

    # ---- dedup ----

    @staticmethod
    def _dedupe_overlapping(matches: list[PIIMatch]) -> list[PIIMatch]:
        """Resolve overlapping spans by keeping higher-confidence matches.

        Two matches overlap if their spans intersect. When that happens,
        we keep the one with higher confidence. Ties broken by longer span
        (more specific match preferred).
        """
        # Sort by start asc, then by (-confidence, -length) for tie-breaking
        sorted_matches = sorted(
            matches,
            key=lambda m: (m.start, -m.confidence, -(m.end - m.start))
        )
        kept: list[PIIMatch] = []
        for m in sorted_matches:
            # Check if m overlaps any kept match
            overlap_idx = None
            for i, k in enumerate(kept):
                if m.start < k.end and m.end > k.start:
                    overlap_idx = i
                    break
            if overlap_idx is None:
                kept.append(m)
            else:
                k = kept[overlap_idx]
                # Replace if m is higher confidence, or same conf but longer span
                if (m.confidence, m.end - m.start) > (k.confidence, k.end - k.start):
                    kept[overlap_idx] = m
        return sorted(kept, key=lambda m: m.start)

    # ---- placeholder mapping ----

    def _placeholder_for(self, pii_type: str, original: str) -> str:
        """Return a consistent placeholder for (type, original) within this session.

        First time we see 'Ramesh' as a NAME → <NAME_1>. Subsequent mentions
        of 'Ramesh' → <NAME_1>. A different name 'Suresh' → <NAME_2>.
        """
        key = (pii_type, original.lower().strip())
        if key not in self._placeholder_map:
            self._counters[pii_type] = self._counters.get(pii_type, 0) + 1
            self._placeholder_map[key] = f"<{pii_type}_{self._counters[pii_type]}>"
        return self._placeholder_map[key]

    # ---- main redaction API ----

    def detect(self, text: str) -> list[PIIMatch]:
        """Run both layers and return the deduped match list."""
        all_matches = self.rules.detect_all(text)
        if self.presidio is not None:
            all_matches.extend(self.presidio.detect(text))
        return self._dedupe_overlapping(all_matches)

    def redact(self, text: str) -> tuple[str, list[PIIMatch]]:
        """Return (redacted_text, list_of_matches_applied).

        Replaces each PII span with its placeholder. Uses the per-session
        placeholder map so repeated PII gets the same placeholder.
        """
        matches = self.detect(text)
        if not matches:
            return text, []

        # Rebuild text from segments — safer than naive replace which would
        # mangle spans if any original text contains another match's substring.
        out_parts: list[str] = []
        cursor = 0
        for m in matches:
            out_parts.append(text[cursor:m.start])
            out_parts.append(self._placeholder_for(m.pii_type, m.original))
            cursor = m.end
        out_parts.append(text[cursor:])
        return "".join(out_parts), matches
        
        


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    sample = (
        "Hi my name is Ramesh, call me at +91 9845012345. "
        "My Aadhaar is 234567890123. From Kolar district. "
        "Field at 12.97, 77.59. Email me at ramesh@gmail.com. "
        "PAN: ABCDE1234F. Bank account 12345678901 in HDFC. "
        "Also my brother Sundar Pichai lives in Bangalore."
    )

    print("\n--- Layer 1 only (rules) ---")
    rules = IndianPIIRules()
    for m in rules.detect_all(sample):
        print(f"  {m.pii_type:14s} | conf={m.confidence:.2f} | src={m.source} | '{m.original}'")

    print("\n--- Full pipeline (rules + Presidio + dedup) ---")
    pipeline = PIIPipeline()
    pipeline.reset_session()
    redacted, matches = pipeline.redact(sample)
    for m in matches:
        print(f"  {m.pii_type:14s} | conf={m.confidence:.2f} | src={m.source} | '{m.original}'")

    print(f"\n--- Redacted text ---\n{redacted}")

    print("\n--- Same session, second redaction (placeholder consistency check) ---")
    text2 = "Ramesh called again from Kolar today."
    redacted2, _ = pipeline.redact(text2)
    print(f"  Input:    {text2}")
    print(f"  Redacted: {redacted2}")
    print("  Expected: Ramesh and Kolar should use the SAME placeholders as above")