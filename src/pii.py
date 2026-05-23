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

    # Words that should NOT be captured as names even after introduction phrases.
    # Catches Hinglish nouns and common English words that follow phrases like "I am".
    NAME_CONTEXT_STOPWORDS = {
        "and", "from", "with", "but", "or", "the", "a", "an",
        "i", "we", "my", "your", "his", "her", "their",
        "looking", "asking", "calling", "writing", "checking",
        "very", "really", "just", "also", "still", "now",
        "mere", "kya", "kab", "hai", "ka", "ki", "ke", "ko", "se",
        "yes", "no", "ok", "okay", "pm", "msp",
    }

    # Inline-case-insensitive intro phrase, but case-SENSITIVE name capture
    # (must start uppercase, followed only by lowercase). This prevents
    # "and", "AND", or all-caps acronyms from being swept into the name span.
    NAME_CONTEXT_RE = re.compile(
        r"(?i:my name is|i am|i'm|this is|name is|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
    )

    def detect_names_by_context(self, text: str) -> list[PIIMatch]:
        """Catch names introduced by phrases like 'my name is X'.

        Safety net for rare names not in the gazetteer. Filters stopwords
        (English connectors + common Hinglish nouns) to avoid over-capture
        on sentences like 'I am looking for' or 'mere tamatar mein'.
        """
        results = []
        for m in self.NAME_CONTEXT_RE.finditer(text):
            name = m.group(1).strip()
            first_token = name.split()[0].lower()

            # Drop matches where the captured token is a stopword
            if first_token in self.NAME_CONTEXT_STOPWORDS:
                continue

            results.append(PIIMatch(
                pii_type="NAME",
                start=m.start(1),
                end=m.end(1),
                original=name,
                confidence=0.85,
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
        matches.extend(self.detect_names_by_context(text))  # ← new
        matches.extend(self.detect_villages(text))
        return sorted(matches, key=lambda m: m.start)


class PresidioPIIDetector:
    """Wraps Microsoft Presidio for NLP-based entity detection.

    Cross-references gazetteers to fix common Presidio mistakes on Indian text:
    - 'Ludhiana' looks like PERSON to spaCy NER → re-tagged as LOCATION
    - Domain vocabulary ('Aadhaar', 'Kisan') filtered as false positives
    """

    ENTITY_MAP = {
        "PERSON": "NAME",
        "LOCATION": "LOCATION",
        "ORGANIZATION": "ORGANIZATION",
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE",
        "CREDIT_CARD": "CREDIT_CARD",
    }

    # Words Presidio mis-tags as PERSON in Indian/agri context.
    PERSON_STOPWORDS = {
        "aadhaar", "pan", "pm-kisan", "pm", "kisan", "msp",
        "kharif", "rabi", "zaid", "rtc", "khasra", "ifsc",
        "hdfc", "sbi", "icici", "axis",
        "mandi", "fasal", "kheti", "paani", "ragi", "bajra",
        # Hinglish function words commonly mis-tagged as PERSON
        "hun", "aur", "hai", "hain", "tha", "thi", "ka", "ki",
        "ke", "ko", "se", "par", "mein", "mere", "meri", "mera",
        "tera", "teri", "mein", "aap", "main", "tu", "yeh", "woh",
        "kya", "kab", "kaise", "kyun", "kahan",
    }

    def __init__(self, villages_gazetteer: set[str] | None = None):
        from presidio_analyzer import AnalyzerEngine
        self._analyzer = AnalyzerEngine()
        # If villages gazetteer provided, use it to correct PERSON→LOCATION mistakes
        self._villages = villages_gazetteer or set()
        logger.info(
            f"Presidio analyzer initialized "
            f"(cross-referencing {len(self._villages)} known villages)"
        )

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
            normalized = matched_text.lower().strip()

            # Filter domain-vocabulary false positives
            if our_type == "NAME":
                # Check if ALL tokens in the match are stopwords → reject as PII
                tokens = matched_text.lower().split()
                if tokens and all(t in self.PERSON_STOPWORDS for t in tokens):
                    continue
                # Also reject if the match is a single stopword
                if matched_text.lower().strip() in self.PERSON_STOPWORDS:
                    continue

            # If Presidio says PERSON but the word is a known village,
            # re-tag as LOCATION. This fixes 'Ludhiana' → NAME mistakes.
            if our_type == "NAME" and normalized in self._villages:
                our_type = "LOCATION"

            matches.append(PIIMatch(
                pii_type=our_type,
                start=r.start,
                end=r.end,
                original=matched_text,
                confidence=float(r.score),
                source="presidio",
            ))
        return matches


class IndicNERDetector:
    """Wraps a multilingual NER model for Devanagari/Indic-script entity detection.

    Uses Davlan/bert-base-multilingual-cased-ner-hrl — an mBERT fine-tuned for
    NER on 10 languages including Hindi. Open-access alternative to
    AI4Bharat's gated IndicNER, with comparable performance on Devanagari.

    Catches PERSON / LOCATION / ORGANIZATION entities in Indic-script text
    that Presidio's English NER and gazetteer-based detection would miss.
    The detector abstracts the underlying model — switching to AI4Bharat's
    IndicNER once access is granted is a one-line model_name change.
    """

    ENTITY_MAP = {
        "PER": "NAME",
        "LOC": "LOCATION",
        "ORG": "ORGANIZATION",
    }

    def __init__(
        self,
        model_name: str = "Davlan/bert-base-multilingual-cased-ner-hrl",
        villages_gazetteer: set[str] | None = None,
    ):
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForTokenClassification.from_pretrained(model_name)
        self._pipe = pipeline(
            "ner",
            model=self._model,
            tokenizer=self._tokenizer,
            aggregation_strategy="simple",
            device=-1,
        )
        self._villages = villages_gazetteer or set()
        logger.info(
            f"Multilingual NER detector initialized ({model_name}) "
            f"with {len(self._villages)} village cross-references"
        )

    def detect(self, text: str) -> list[PIIMatch]:
        if not text or not text.strip():
            return []
        try:
            results = self._pipe(text)
        except Exception as e:
            logger.warning(f"Multilingual NER failed on text snippet: {e}")
            return []
        matches: list[PIIMatch] = []
        for r in results:
            label = r.get("entity_group", "").upper()
            our_type = self.ENTITY_MAP.get(label)
            if our_type is None:
                continue
            matched_text = text[r["start"]:r["end"]]
            normalized = matched_text.lower().strip()

            # Cross-reference: if NER says PERSON but the token is a known village,
            # re-tag as LOCATION. Same correction we apply to Presidio.
            if our_type == "NAME" and normalized in self._villages:
                our_type = "LOCATION"

            matches.append(PIIMatch(
                pii_type=our_type,
                start=int(r["start"]),
                end=int(r["end"]),
                original=matched_text,
                confidence=float(r["score"]),
                source="multilingual_ner",
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

    def __init__(self, use_presidio: bool = True, use_indicner: bool = False):
        self.rules = IndianPIIRules()
        self.presidio = PresidioPIIDetector(villages_gazetteer=self.rules.villages) if use_presidio else None
        self.indicner = IndicNERDetector(villages_gazetteer=self.rules.villages) if use_indicner else None
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

    def detect(self, text: str, language_hint: str | None = None) -> list[PIIMatch]:
        """Run all detection layers and return the deduped match list.

        Args:
            text: the string to scan for PII.
            language_hint: one of "hi", "code_mixed", "en", or None.
                When "en", the Indic NER layer is skipped (it adds latency
                and produces false positives on pure-English text).
        """
        all_matches = self.rules.detect_all(text)
        if self.presidio is not None:
            all_matches.extend(self.presidio.detect(text))
        if self.indicner is not None and language_hint != "en":
            all_matches.extend(self.indicner.detect(text))
        return self._dedupe_overlapping(all_matches)

    def redact(self, text: str, language_hint: str | None = None) -> tuple[str, list[PIIMatch]]:
        """Return (redacted_text, list_of_matches_applied).

        Replaces each PII span with its placeholder. Uses the per-session
        placeholder map so repeated PII gets the same placeholder.

        Args:
            text: the string to redact.
            language_hint: forwarded to detect() to control Indic NER usage.
        """
        matches = self.detect(text, language_hint=language_hint)
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

    # ---- trajectory-level redaction ----

    def redact_trajectory(self, trajectory_dict: dict) -> tuple[dict, list[dict]]:
        """Redact all PII in every event of a trajectory.

        Resets the placeholder map at the start so each trajectory has its own
        isolated placeholder namespace (privacy isolation across sessions).

        Args:
            trajectory_dict: a trajectory as a dict (loaded from JSONL)

        Returns:
            (redacted_trajectory_dict, audit_records)
            audit_records is a list of per-event redactions for the audit log:
                [{"event_id": ..., "field": ..., "original": ..., "redacted": ...,
                  "matches": [...]}, ...]
            Empty if no PII was found in the trajectory.
        """
        self.reset_session()
        language_hint = trajectory_dict.get("language_mix")
        audit_records: list[dict] = []

        for event in trajectory_dict["events"]:
            # Redact the content field (user/assistant/system messages)
            if event.get("content"):
                redacted, matches = self.redact(event["content"], language_hint=language_hint)
                if matches:
                    audit_records.append({
                        "event_id": event["event_id"],
                        "field": "content",
                        "original": event["content"],
                        "redacted": redacted,
                        "matches": [
                            {
                                "type": m.pii_type,
                                "confidence": m.confidence,
                                "source": m.source,
                                "original": m.original,
                            }
                            for m in matches
                        ],
                    })
                    event["content"] = redacted

            # Redact tool_args (a dict of string values mostly)
            if event.get("tool_args"):
                event["tool_args"], arg_audits = self._redact_dict_strings(
                    event["tool_args"], event["event_id"], "tool_args"
                )
                audit_records.extend(arg_audits)

            # Redact tool_output (also a dict)
            if event.get("tool_output"):
                event["tool_output"], out_audits = self._redact_dict_strings(
                    event["tool_output"], event["event_id"], "tool_output"
                )
                audit_records.extend(out_audits)

        return trajectory_dict, audit_records

    def _redact_dict_strings(
        self,
        d: dict,
        event_id: str,
        field_name: str,
    ) -> tuple[dict, list[dict]]:
        """Recursively walk a dict and redact any string values.

        We only redact string leaves — numbers, bools, and nested dicts pass through
        (nested dicts are recursed into). This prevents accidental corruption of
        structured fields like `price_per_quintal: 2150`.
        """
        audits: list[dict] = []
        new_d: dict = {}
        for key, value in d.items():
            if isinstance(value, str):
                redacted, matches = self.redact(value)
                if matches:
                    audits.append({
                        "event_id": event_id,
                        "field": f"{field_name}.{key}",
                        "original": value,
                        "redacted": redacted,
                        "matches": [
                            {
                                "type": m.pii_type,
                                "confidence": m.confidence,
                                "source": m.source,
                                "original": m.original,
                            }
                            for m in matches
                        ],
                    })
                new_d[key] = redacted
            elif isinstance(value, dict):
                new_d[key], nested_audits = self._redact_dict_strings(
                    value, event_id, f"{field_name}.{key}"
                )
                audits.extend(nested_audits)
            elif isinstance(value, list):
                # Lists may contain dicts (e.g. weather forecast). Recurse into them.
                new_list = []
                for item in value:
                    if isinstance(item, dict):
                        red_item, nested_audits = self._redact_dict_strings(
                            item, event_id, f"{field_name}.{key}[]"
                        )
                        new_list.append(red_item)
                        audits.extend(nested_audits)
                    elif isinstance(item, str):
                        redacted, matches = self.redact(item)
                        if matches:
                            audits.append({
                                "event_id": event_id,
                                "field": f"{field_name}.{key}[]",
                                "original": item,
                                "redacted": redacted,
                                "matches": [
                                    {"type": m.pii_type, "confidence": m.confidence,
                                     "source": m.source, "original": m.original}
                                    for m in matches
                                ],
                            })
                        new_list.append(redacted)
                    else:
                        new_list.append(item)
                new_d[key] = new_list
            else:
                # numbers, bools, None — pass through unchanged
                new_d[key] = value
        return new_d, audits


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