"""Deterministic validation of what the LLM claims against what the evidence can carry.

The LLM writes two kinds of text that reach a human: a short ``reasoning`` for each ranked cause
and an operator-facing explanation. This module decides, without any model, whether each
sentence claims more than the evidence supports.

Two steps, both rule-based and bounded:

1. **Classify the language** of a sentence with fixed lexicons: causal ("caused", "due to",
   "led to"), certain ("proves", "definitely"), hedged ("consistent with", "may"), associative
   ("alongside", "correlated with"), or a plain assertion. Negated causation ("does not establish
   causality") is recognised and is not a causal claim.
2. **Compare the strength the wording needs with the strength of the evidence available**:

   ============================ ===============================================================
   OBSERVATION                  a monitor measured something (drift, a missing-value rate)
   ASSOCIATION                  a monitor measured a relationship (feature-label association)
   MECHANISM                    a diagnostic showed *how* something changed (constant-factor fit,
                                artifact hash differs)
   INTERVENTION                 the sandbox applied the corresponding fix and the targeted
                                findings cleared
   ============================ ===============================================================

   A fact, an association or a hypothesis needs relevant OBSERVATION-grade evidence. A causal or
   certain claim needs INTERVENTION-grade evidence for the same cause. Co-occurrence of a drift
   alert and a performance drop is two observations; it never reaches INTERVENTION.

Verdicts: ``supported``; ``rejected`` (the text is replaced or the item dropped); ``escalated``
(causal wording with MECHANISM-grade but not INTERVENTION-grade evidence: the wording is replaced
*and* the case is flagged for the human reviewer).

Limits, stated once here and in ``docs/evidence-and-claim-validation.md``: a lexicon recognises
phrasings, not meaning. A paraphrase outside it is classified as a plain assertion and is checked
only for relevant evidence. The validator can only tighten what the LLM said; it never raises a
confidence, adds evidence, or changes a deterministic result.
"""

from __future__ import annotations

import re
from enum import IntEnum, StrEnum

from pydantic import BaseModel, Field

from aegis.schemas.diagnosis import Diagnosis, Evidence, RootCause
from aegis.schemas.remediation import RemediationProposal, ValidationResult

MAX_TEXT_CHARS = 2000
MAX_SENTENCES = 12


class ClaimCategory(StrEnum):
    OBSERVED_FACT = "observed_fact"
    STATISTICAL_ASSOCIATION = "statistical_association"
    HYPOTHESIS = "hypothesis"
    CAUSAL_CLAIM = "causal_claim"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CONTRADICTORY_CLAIM = "contradictory_claim"


class ClaimLanguage(StrEnum):
    CAUSAL = "causal"
    CERTAIN = "certain"
    HEDGED = "hedged"
    ASSOCIATIVE = "associative"
    ASSERTIVE = "assertive"
    NEGATED_CAUSAL = "negated_causal"
    PROCEDURAL = "procedural"


class EvidenceStrength(IntEnum):
    NONE = 0
    OBSERVATION = 1
    ASSOCIATION = 2
    MECHANISM = 3
    INTERVENTION = 4


class ClaimVerdict(StrEnum):
    SUPPORTED = "supported"
    REJECTED = "rejected"
    ESCALATED = "escalated"


class Claim(BaseModel):
    """One validated statement. Everything needed to audit the verdict is on the record."""

    claim_id: str
    origin: str  # "llm:rank" | "llm:explain"
    text: str
    language: ClaimLanguage
    category: ClaimCategory
    subject_causes: list[str] = Field(default_factory=list)
    subject_features: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    valid_evidence_ids: list[str] = Field(default_factory=list)
    fabricated_evidence_ids: list[str] = Field(default_factory=list)
    irrelevant_evidence_ids: list[str] = Field(default_factory=list)
    untrusted_evidence_ids: list[str] = Field(default_factory=list)
    evidence_strength: str = EvidenceStrength.NONE.name
    required_strength: str = EvidenceStrength.OBSERVATION.name
    verdict: ClaimVerdict
    reasons: list[str] = Field(default_factory=list)


class Intervention(BaseModel):
    """A sandbox validation, reduced to what a causal claim may lean on."""

    target_root_cause: str | None
    features: list[str] = Field(default_factory=list)
    passed: bool
    effect_checks_passed: bool

    @classmethod
    def from_validation(
        cls, proposal: RemediationProposal | dict | None, validation: ValidationResult | dict | None
    ) -> Intervention | None:
        if not proposal or not validation:
            return None
        proposal = (
            proposal
            if isinstance(proposal, RemediationProposal)
            else RemediationProposal.model_validate(proposal)
        )
        validation = (
            validation
            if isinstance(validation, ValidationResult)
            else ValidationResult.model_validate(validation)
        )
        effects = [c for c in validation.checks if c.category == "effect"]
        feature = proposal.parameters.get("feature")
        return cls(
            target_root_cause=proposal.target_root_cause,
            features=[feature] if isinstance(feature, str) else [],
            passed=validation.passed,
            effect_checks_passed=bool(effects) and all(c.passed for c in effects),
        )

    def supports(self, cause: str) -> bool:
        return self.passed and self.effect_checks_passed and self.target_root_cause == cause


COUNTERS = (
    "claims_evaluated",
    "causal_claims_generated",
    "causal_claims_supported",
    "causal_claims_rejected",
    "causal_claims_escalated",
    "unsupported_claims_generated",
    "unsupported_claims_rejected",
    "invalid_cause_labels_rejected",
    "contradictory_claims_generated",
    "contradictory_claims_rejected",
    "fabricated_evidence_ids_generated",
    "fabricated_evidence_ids_stripped",
    "irrelevant_evidence_ids_stripped",
    "untrusted_evidence_ids_stripped",
)
ESCAPE_COUNTERS = (
    "causal_claims_escaped",
    "unsupported_claims_escaped",
    "fabricated_evidence_ids_escaped",
)


def empty_counters() -> dict[str, int]:
    return dict.fromkeys(COUNTERS, 0)


def add_counters(*parts: dict[str, int] | None) -> dict[str, int]:
    total = empty_counters()
    for part in parts:
        for key, value in (part or {}).items():
            if key in total or key in ESCAPE_COUNTERS:
                total[key] = total.get(key, 0) + int(value)
    return total


# --------------------------------------------------------------------------- language
_NEGATED = re.compile(
    r"\bcaus\w+\s+(?:is|was|has|remains)\s+(?:not|un)\s*(?:yet\s+)?(?:been\s+)?"
    r"(?:established|proven|confirmed|shown|demonstrated|known)\b|"
    r"\b(?:does|do|did)\s+not\s+(?:necessarily\s+|by\s+itself\s+|alone\s+)?"
    r"(?:cause|establish|prove|confirm|explain|show|demonstrate|imply|mean)\w*(?:\s+caus\w+)?"
    r"|\b(?:is|was|are|were)\s+not\s+(?:necessarily\s+|yet\s+|been\s+)?"
    r"(?:caused|due|the\s+(?:root\s+)?cause|established|proven|a\s+result|responsible|confirmed)\w*"
    r"|\bnot\s+(?:yet\s+)?(?:been\s+)?(?:established|proven|confirmed|demonstrated)(?:\s+caus\w+)?"
    r"|\b(?:cannot|can\s*not|can't)\s+(?:be\s+)?(?:conclude|establish|confirm|prove|say)\w*"
    r"|\bwithout\s+establishing\b"
    r"|\bno\s+(?:evidence|proof)\s+(?:that|of)\b.*$"
    r"|\bnot\s+established\s+caus\w+",
    re.IGNORECASE,
)
_EXPLICIT_CAUSE = re.compile(
    r"\bthe\s+(?:root\s+)?cause\s+(?:is|was|of)\b|\bis\s+the\s+(?:root\s+)?cause\b|\broot\s+cause\s+is\b",
    re.IGNORECASE,
)
_ROOT_CAUSE_NOUN = re.compile(r"\broot[\s-]causes?\b", re.IGNORECASE)
_CAUSAL = re.compile(
    r"\bcaus(?:e|ed|es|ing|al|ally|ality|ation)\b|\bbecause\b|\bdue\s+to\b|\bowing\s+to\b"
    r"|\bled\s+to\b|\bleads?\s+to\b|\bleading\s+to\b|\bresult(?:s|ed|ing)?\s+(?:in|from)\b"
    r"|\bas\s+a\s+result\b|\bresponsible\s+for\b|\btriggered\s+(?:by|the|a)\b|\bdriven\s+by\b|\bdrove\b"
    r"|\bexplains?\s+(?:the|why|this)\b|\battribut(?:ed|able)\s+to\b|\bstem(?:s|med|ming)?\s+from\b"
    r"|\bbrought\s+about\b|\bconsequently\b"
    r"|\bdegraded\s+the\b|\bis\s+why\b|\bthe\s+reason\b|\bculprit\b"
    r"|\bto\s+blame\b|\bby\s+virtue\s+of\b|\bon\s+account\s+of\b",
    re.IGNORECASE,
)
_CERTAIN = re.compile(
    r"\bproves?\b|\bproven\b|\bproof\b|\bdefinitely\b|\bcertainly\b|\bundoubtedly\b|\bwithout\s+(?:a\s+)?doubt\b"
    r"|\bguarantee[sd]?\b|\bconfirm(?:s|ed)\s+(?:that|the)\b|\bconclusively\b|\bestablishe[sd]\s+that\b"
    r"|\bclearly\s+(?:shows?|demonstrates?)\b|\bunquestionably\b|\bno\s+doubt\b|\bfor\s+certain\b",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"\bconsistent\s+with\b|\bmay\b|\bmight\b|\bcould\b|\bpossibl[ey]\b|\blikely\b|\bprobabl[ey]\b"
    r"|\bappears?\s+to\b|\bseems?\b|\bsuggests?\b|\bhypothes[ie]s\b|\bcandidates?\b|\bevidence\s+fit\b"
    r"|\bplausibl[ey]\b|\bpotential(?:ly)?\b|\bindicat(?:es?|ive)\b|\bcompatible\s+with\b|\bpoints?\s+to\b",
    re.IGNORECASE,
)
_ASSOCIATION = re.compile(
    r"\balongside\b|\bassociated\s+with\b|\bcorrelat\w+\b|\bcoincid\w+\b|\bat\s+the\s+same\s+time\b"
    r"|\btogether\s+with\b|\baccompanied\s+by\b|\bco-?occurr?\w*\b|\bconcurrent\w*\b|\balong\s+with\b",
    re.IGNORECASE,
)
_PERFORMANCE = re.compile(
    r"\bperformance\b|\baccuracy\b|\bprecision\b|\brecall\b|\bf1\b|\bauc\b|\berror\s+rate\b|\bdegrad\w+\b",
    re.IGNORECASE,
)
_DENIES_ANOMALY = re.compile(
    r"\bno\s+(?:drift|anomal\w+|issue|change|shift|problem)s?\b|\b(?:is|are|looks?|remains?)\s+normal\b"
    r"|\bunchanged\b|\bnot\s+(?:drifted|changed|shifted|affected)\b|\bwithin\s+tolerance\b",
    re.IGNORECASE,
)
_PROCEDURAL = re.compile(
    r"^\W*(?:proposed\s+action|recommended\s+action|recommendation|action|sandbox(?:\s+validation)?"
    r"|validation|policy(?:\s+(?:outcome|decision))?|status|approval)\b\s*[:\-]?",
    re.IGNORECASE,
)
_CAUSE_WORD = re.compile(r"\bcaus(?:e|ed|es|ing|al|ally|ality|ation)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?;])\s+|\n+")

#: Checks that can speak to each root cause (from the rule engine's own definitions).
PERF_CHECKS = frozenset(
    {"accuracy_drop", "precision_drop", "recall_drop", "f1_drop", "roc_auc_drop"}
)
RELEVANT_CHECKS: dict[RootCause, frozenset[str]] = {
    RootCause.NUMERICAL_SHIFT: frozenset(
        {"numeric_drift", "range_violation_rate", "feature_pattern"}
    ),
    RootCause.SCALING_ERROR: frozenset(
        {"numeric_drift", "range_violation_rate", "feature_pattern"}
    ),
    RootCause.CATEGORICAL_SHIFT: frozenset(
        {"categorical_drift", "unexpected_category_rate", "cardinality_change", "feature_pattern"}
    ),
    RootCause.MISSING_VALUES: frozenset({"missing_rate"}),
    RootCause.DUPLICATE_RECORDS: frozenset({"duplicate_rate"}),
    RootCause.SCHEMA_CHANGE: frozenset(
        {"schema_missing_columns", "schema_type_mismatch", "schema_unexpected_columns"}
    ),
    RootCause.FEATURE_LEAKAGE: frozenset(
        {"label_association_gain", "numeric_drift", "categorical_drift"}
    ),
    RootCause.CONCEPT_DRIFT: frozenset({"label_association_reversal", "adwin", "page_hinkley"}),
    RootCause.LABEL_DELAY: frozenset({"label_availability"}),
    RootCause.ARTIFACT_MISMATCH: frozenset({"artifact_hash", "prediction_distribution"}),
}
#: Tool outputs about model behaviour in general (not tied to one root cause).
_GENERAL_DIAGNOSTICS = frozenset({"feature_importance", "error_slices", "version_comparison"})
_ASSOCIATION_CHECKS = frozenset({"label_association_gain", "label_association_reversal"})
_MECHANISM_CHECKS = frozenset({"feature_pattern", "artifact_hash", "version_comparison"})
_TRUSTED_SOURCES = ("monitoring:", "tool:")
_UNTRUSTED_TOOLS = ("tool:retrieve_runbook",)

_CAUSE_ALIASES: dict[RootCause, tuple[str, ...]] = {
    RootCause.NUMERICAL_SHIFT: ("numerical shift", "numeric shift", "mean shift"),
    RootCause.CATEGORICAL_SHIFT: ("categorical shift", "category shift"),
    RootCause.MISSING_VALUES: ("missing values", "missing-value"),
    RootCause.SCALING_ERROR: ("scaling error", "unit error", "scale error"),
    RootCause.CONCEPT_DRIFT: ("concept drift",),
    RootCause.DUPLICATE_RECORDS: ("duplicate records", "duplicate rows", "duplicates"),
    RootCause.FEATURE_LEAKAGE: ("feature leakage", "label leakage", "leakage"),
    RootCause.SCHEMA_CHANGE: ("schema change",),
    RootCause.LABEL_DELAY: ("label delay", "delayed labels"),
    RootCause.ARTIFACT_MISMATCH: ("artifact mismatch", "model mismatch"),
}


def split_sentences(text: str) -> list[str]:
    text = (text or "")[:MAX_TEXT_CHARS]
    return [s.strip() for s in _SENTENCE.split(text) if s and s.strip()][:MAX_SENTENCES]


def classify_language(sentence: str) -> ClaimLanguage:
    """Which kind of statement the wording makes. Purely lexical and deterministic."""
    negated = bool(_NEGATED.search(sentence))
    remainder = _NEGATED.sub(" ", sentence)
    explicit = bool(_EXPLICIT_CAUSE.search(remainder))
    remainder = _ROOT_CAUSE_NOUN.sub(" ", remainder)  # the noun phrase alone asserts nothing
    if _CERTAIN.search(remainder):
        return ClaimLanguage.CERTAIN
    # A sentence labelled as one of the workflow's own decided facts ("Proposed action: ...",
    # "Sandbox validation: ...") states why the system decided something, not what caused the
    # incident. "due to" / "because" there is a rationale. An explicit cause-word is still a
    # causal claim, so nothing can be smuggled through behind such a label.
    if _PROCEDURAL.search(sentence) and not explicit and not _CAUSE_WORD.search(remainder):
        return ClaimLanguage.PROCEDURAL
    causal = explicit or bool(_CAUSAL.search(remainder))
    hedged = bool(_HEDGE.search(remainder))
    if causal and not hedged:
        return ClaimLanguage.CAUSAL
    if causal or hedged:
        return ClaimLanguage.HEDGED  # "may have caused": a hypothesis about a cause
    if negated:
        return ClaimLanguage.NEGATED_CAUSAL
    if _ASSOCIATION.search(remainder):
        return ClaimLanguage.ASSOCIATIVE
    return ClaimLanguage.ASSERTIVE


_LANGUAGE_PRIORITY = (
    ClaimLanguage.CERTAIN,
    ClaimLanguage.CAUSAL,
    ClaimLanguage.HEDGED,
    ClaimLanguage.ASSOCIATIVE,
    ClaimLanguage.NEGATED_CAUSAL,
    ClaimLanguage.PROCEDURAL,
    ClaimLanguage.ASSERTIVE,
)


def strongest_language(text: str) -> ClaimLanguage:
    """The most demanding language used in any sentence of ``text``.

    A hedge in one sentence does not soften a causal assertion in another.
    """
    found = {classify_language(s) for s in split_sentences(text)}
    return next((lang for lang in _LANGUAGE_PRIORITY if lang in found), ClaimLanguage.ASSERTIVE)


def mentioned_causes(text: str) -> list[RootCause]:
    lowered = text.lower().replace("_", " ")
    return [
        cause
        for cause, aliases in _CAUSE_ALIASES.items()
        if cause.value.replace("_", " ") in lowered or any(a in lowered for a in aliases)
    ]


def mentioned_features(text: str, evidence: list[Evidence]) -> list[str]:
    features = sorted({e.feature for e in evidence if e.feature}, key=len, reverse=True)
    lowered = text.lower()
    return [f for f in features if re.search(rf"(?<![\w]){re.escape(f.lower())}(?![\w])", lowered)]


# --------------------------------------------------------------------------- evidence
def is_trusted(evidence: Evidence) -> bool:
    """Provenance: produced by a monitor or a diagnostic tool, not by retrieved text."""
    return evidence.source.startswith(_TRUSTED_SOURCES) and not evidence.source.startswith(
        _UNTRUSTED_TOOLS
    )


def strength_of(evidence: Evidence) -> EvidenceStrength:
    if not is_trusted(evidence):
        return EvidenceStrength.NONE
    check = evidence.data.get("check")
    if check in _MECHANISM_CHECKS and evidence.data.get("polarity") == "anomaly":
        return EvidenceStrength.MECHANISM
    if check in _ASSOCIATION_CHECKS and evidence.data.get("polarity") == "anomaly":
        return EvidenceStrength.ASSOCIATION
    return EvidenceStrength.OBSERVATION


def is_relevant(
    evidence: Evidence,
    causes: list[RootCause],
    features: list[str],
    rule_ids: frozenset[str] = frozenset(),
) -> bool:
    """Does this evidence speak to the thing the claim is about?

    With no identifiable subject, relevance cannot be judged and is not held against the claim.
    """
    if not causes and not features:
        return True
    if evidence.evidence_id in rule_ids:
        return True  # the rule engine itself ties this evidence to the cause
    check = str(evidence.data.get("check") or "")
    if check in PERF_CHECKS or check in _GENERAL_DIAGNOSTICS or check.startswith("no_"):
        # impact evidence, model-behaviour diagnostics and "what did not happen" bear on
        # every explanation
        return True
    if features and evidence.feature in features:
        return True
    return any(check in RELEVANT_CHECKS.get(cause, frozenset()) for cause in causes)


def _is_anomaly(evidence: Evidence) -> bool:
    return evidence.data.get("polarity") == "anomaly"


def evaluate_claim(
    *,
    claim_id: str,
    origin: str,
    text: str,
    evidence: list[Evidence],
    cited_ids: list[str] | None,
    causes: list[RootCause] | None = None,
    rule_diagnosis: Diagnosis | None = None,
    intervention: Intervention | None = None,
    whole_text: bool = False,
) -> Claim:
    """Validate one statement.

    ``whole_text`` treats ``text`` as several sentences making one claim (a ranking item's
    reasoning) and judges it by its most demanding sentence.

    ``cited_ids`` is the evidence the statement itself points at (ranking items cite ids);
    ``None`` means the statement cites nothing explicitly (explanation prose), in which case
    all evidence of the investigation is the pool it is checked against.
    """
    known = {e.evidence_id: e for e in evidence}
    language = strongest_language(text) if whole_text else classify_language(text)
    subject_causes = list(dict.fromkeys([*(causes or []), *mentioned_causes(text)]))
    features = mentioned_features(text, evidence)
    rule_support: set[str] = set()
    rule_contra: set[str] = set()
    for hypothesis in rule_diagnosis.hypotheses if rule_diagnosis else []:
        if hypothesis.root_cause in subject_causes:
            rule_support |= set(hypothesis.supporting_evidence_ids)
            rule_contra |= set(hypothesis.contradicting_evidence_ids)
    rule_ids = frozenset(rule_support | rule_contra)

    cited = list(dict.fromkeys(cited_ids)) if cited_ids is not None else list(known)
    fabricated = [i for i in cited if i not in known] if cited_ids is not None else []
    existing = [known[i] for i in cited if i in known]
    untrusted = [e.evidence_id for e in existing if not is_trusted(e)]
    trusted = [e for e in existing if is_trusted(e)]
    irrelevant = [
        e.evidence_id for e in trusted if not is_relevant(e, subject_causes, features, rule_ids)
    ]
    valid = [e for e in trusted if e.evidence_id not in irrelevant]
    anomalies = [e for e in valid if _is_anomaly(e)]
    strength = max((strength_of(e) for e in anomalies), default=EvidenceStrength.NONE)
    if intervention and any(intervention.supports(c.value) for c in subject_causes):
        strength = EvidenceStrength.INTERVENTION

    reasons: list[str] = []
    if fabricated:
        reasons.append(f"cites evidence ids that do not exist: {fabricated}")
    if untrusted:
        reasons.append(f"cites evidence without deterministic provenance: {untrusted}")
    if irrelevant:
        reasons.append(f"cites evidence unrelated to the claim's subject: {irrelevant}")

    # Contradiction: leaning on evidence the rule engine counts against the same cause, or
    # denying an anomaly that the cited/known evidence records for the named feature.
    contradicts = cited_ids is not None and bool(set(cited) & (rule_contra - rule_support))
    if contradicts:
        reasons.append("cites as support evidence that contradicts this cause")
    if (
        features
        and _DENIES_ANOMALY.search(text)
        and any(_is_anomaly(e) and e.feature in features for e in known.values())
    ):
        contradicts = True
        reasons.append("denies an anomaly that the evidence records for the named feature")

    needs_performance = language is ClaimLanguage.ASSOCIATIVE and bool(_PERFORMANCE.search(text))
    has_performance = any(
        e.data.get("check") in PERF_CHECKS and _is_anomaly(e) for e in known.values()
    )

    required = EvidenceStrength.OBSERVATION
    if language in (ClaimLanguage.CAUSAL, ClaimLanguage.CERTAIN):
        category, required = ClaimCategory.CAUSAL_CLAIM, EvidenceStrength.INTERVENTION
    elif language in (ClaimLanguage.HEDGED, ClaimLanguage.NEGATED_CAUSAL):
        category = ClaimCategory.HYPOTHESIS
    elif language is ClaimLanguage.ASSOCIATIVE:
        category = ClaimCategory.STATISTICAL_ASSOCIATION
    else:
        category = ClaimCategory.OBSERVED_FACT

    if contradicts:
        category, verdict = ClaimCategory.CONTRADICTORY_CLAIM, ClaimVerdict.REJECTED
    elif category is ClaimCategory.CAUSAL_CLAIM:
        if strength >= EvidenceStrength.INTERVENTION:
            verdict = ClaimVerdict.SUPPORTED
            reasons.append("a passed sandbox intervention for the same cause supports the wording")
        elif (
            language is ClaimLanguage.CAUSAL  # certainty is never softened to a review flag
            and subject_causes  # the claim must name the cause the evidence is about
            and strength >= EvidenceStrength.MECHANISM
            and (has_performance or not _PERFORMANCE.search(text))
        ):
            verdict = ClaimVerdict.ESCALATED
            reasons.append(
                "causal wording with mechanism-level evidence only: no intervention shows that "
                "reversing the change removes the effect; wording replaced, flagged for review"
            )
        else:
            verdict = ClaimVerdict.REJECTED
            reasons.append(
                f"causal wording needs INTERVENTION evidence; available: {strength.name}. "
                "Detected anomalies establish association, not causation"
            )
    elif (
        not anomalies
        and subject_causes
        # A ranking item (it cites ids) proposes its cause as a hypothesis whatever its wording,
        # so it needs anomaly evidence even when the text is a denial ("no evidence of ...").
        # Only free prose may state that causality is not established without citing anything.
        and (cited_ids is not None or language is not ClaimLanguage.NEGATED_CAUSAL)
    ):
        category, verdict = ClaimCategory.UNSUPPORTED_CLAIM, ClaimVerdict.REJECTED
        reasons.append("no relevant anomaly evidence supports this statement")
    elif needs_performance and not has_performance:
        category, verdict = ClaimCategory.UNSUPPORTED_CLAIM, ClaimVerdict.REJECTED
        reasons.append("refers to a performance change that no monitor recorded")
    else:
        verdict = ClaimVerdict.SUPPORTED

    return Claim(
        claim_id=claim_id,
        origin=origin,
        text=text[:600],
        language=language,
        category=category,
        subject_causes=[c.value for c in subject_causes],
        subject_features=features,
        cited_evidence_ids=cited if cited_ids is not None else [],
        valid_evidence_ids=[e.evidence_id for e in valid],
        fabricated_evidence_ids=fabricated,
        # Only ids the statement itself cited can be "stripped"; prose that cites nothing is
        # checked against the whole pool, and unrelated evidence in the pool is not its fault.
        irrelevant_evidence_ids=irrelevant if cited_ids is not None else [],
        untrusted_evidence_ids=untrusted if cited_ids is not None else [],
        evidence_strength=strength.name,
        required_strength=required.name,
        verdict=verdict,
        reasons=reasons,
    )


def count(claims: list[Claim]) -> dict[str, int]:
    """Counters for a batch of validated claims (what was generated and what happened to it)."""
    c = empty_counters()
    for claim in claims:
        c["claims_evaluated"] += 1
        c["fabricated_evidence_ids_generated"] += len(claim.fabricated_evidence_ids)
        c["fabricated_evidence_ids_stripped"] += len(claim.fabricated_evidence_ids)
        c["irrelevant_evidence_ids_stripped"] += len(claim.irrelevant_evidence_ids)
        c["untrusted_evidence_ids_stripped"] += len(claim.untrusted_evidence_ids)
        if claim.category is ClaimCategory.CAUSAL_CLAIM:
            c["causal_claims_generated"] += 1
            c[f"causal_claims_{claim.verdict.value}"] += 1
        elif claim.category is ClaimCategory.UNSUPPORTED_CLAIM:
            c["unsupported_claims_generated"] += 1
            c["unsupported_claims_rejected"] += 1
        elif claim.category is ClaimCategory.CONTRADICTORY_CLAIM:
            c["contradictory_claims_generated"] += 1
            c["contradictory_claims_rejected"] += 1
    return c


def hedged_rewrite(causes: list[str], features: list[str]) -> str:
    """The deterministic sentence that replaces a rejected or escalated causal statement."""
    cause = causes[0].replace("_", " ") if causes else "the leading hypothesis"
    where = f" in {', '.join(features[:3])}" if features else ""
    return (
        f"The cited evidence is consistent with {cause}{where}. "
        "The available evidence does not establish causality."
    )


def validate_text(
    text: str,
    *,
    origin: str,
    id_prefix: str,
    evidence: list[Evidence],
    rule_diagnosis: Diagnosis | None = None,
    intervention: Intervention | None = None,
    causes: list[RootCause] | None = None,
    cited_ids: list[str] | None = None,
) -> list[Claim]:
    """Validate every sentence of a piece of LLM-written text."""
    return [
        evaluate_claim(
            claim_id=f"{id_prefix}-{index}",
            origin=origin,
            text=sentence,
            evidence=evidence,
            cited_ids=cited_ids,
            causes=causes,
            rule_diagnosis=rule_diagnosis,
            intervention=intervention,
        )
        for index, sentence in enumerate(split_sentences(text), start=1)
    ]


def audit_final_output(
    *,
    diagnosis: Diagnosis | None,
    explanation: dict | None,
    evidence: list[Evidence],
    rule_diagnosis: Diagnosis | None = None,
    intervention: Intervention | None = None,
) -> dict[str, int]:
    """Re-scan what actually leaves the system and count anything that should not be there.

    Only LLM-authored text is scanned (hypotheses ranked by the LLM, an explanation whose source
    is the LLM). These counters measure escapes *that this validator can recognise*; they are a
    consistency check on the pipeline, not a proof that no over-claim got through.
    """
    known = {e.evidence_id: e for e in evidence}
    escaped = dict.fromkeys(ESCAPE_COUNTERS, 0)
    for hypothesis in diagnosis.hypotheses if diagnosis else []:
        ids = [*hypothesis.supporting_evidence_ids, *hypothesis.contradicting_evidence_ids]
        escaped["fabricated_evidence_ids_escaped"] += sum(i not in known for i in ids)
        if hypothesis.ranked_by != "llm":
            continue
        support = [known[i] for i in hypothesis.supporting_evidence_ids if i in known]
        if not any(_is_anomaly(e) for e in support):
            escaped["unsupported_claims_escaped"] += 1
        for claim in validate_text(
            hypothesis.reasoning,
            origin="final:hypothesis",
            id_prefix="AUD",
            evidence=evidence,
            rule_diagnosis=rule_diagnosis,
            intervention=intervention,
            causes=[hypothesis.root_cause],
        ):
            if (
                claim.category is ClaimCategory.CAUSAL_CLAIM
                and claim.verdict is not ClaimVerdict.SUPPORTED
            ):
                escaped["causal_claims_escaped"] += 1
    if explanation and str(explanation.get("source", "")).startswith("llm"):
        text = " ".join(
            [str(explanation.get("summary", "")), *map(str, explanation.get("operator_checks", []))]
        )
        for claim in validate_text(
            text,
            origin="final:explanation",
            id_prefix="AUD",
            evidence=evidence,
            rule_diagnosis=rule_diagnosis,
            intervention=intervention,
        ):
            if (
                claim.category is ClaimCategory.CAUSAL_CLAIM
                and claim.verdict is not ClaimVerdict.SUPPORTED
            ):
                escaped["causal_claims_escaped"] += 1
    return escaped
