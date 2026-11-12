# Evidence and claim validation

What the system checks about the text a language model writes, what it does not check, and what
the counters in the benchmark report mean. Code: `aegis/agents/claims.py`,
`aegis/agents/llm_tasks.py`, `aegis/agents/graph.py`.

## What existed before, stated accurately

Before this layer, validation of model output was two things:

1. **Evidence references**: cited ids had to exist, and a ranked cause needed at least one cited
   anomaly. Invented ids were stripped.
2. **One regular expression** over the `reasoning` string and the explanation, matching phrases
   such as "caused by", "proves", "definitely".

The old report line "causal over-claims caught: 0" meant: that expression matched none of the
ranking `reasoning` strings in the run. It did not mean the model made no causal claims
("due to", "led to" and "because of" were not in the expression), it did not cover explanations
(the benchmark had them switched off), and nothing measured what got through. The raw text was
not stored, so that run cannot be re-examined. Its claim metrics are reported as unavailable.

## What exists now

A deterministic validator. No model takes part in it, and it can only tighten what the model
said: it never raises a confidence, adds evidence, or changes a monitoring result, a policy
decision or a validation outcome.

### Step 1: classify the wording

Each sentence is classified by fixed lexicons:

| Language | Examples | Becomes |
|---|---|---|
| causal | caused, due to, led to, because, responsible for, the root cause is, stems from, resulted in | causal claim |
| certain | proves, definitely, certainly, conclusively, without a doubt, guarantees | causal claim |
| hedged | consistent with, may, might, likely, suggests, hypothesis, plausible; also "may have caused" | hypothesis |
| associative | alongside, correlated with, coincides with, at the same time | statistical association |
| negated causal | does not establish causality, no evidence that ..., causation is not established | hypothesis (accepted) |
| assertive | none of the above | observed fact |

Certainty overrides a hedge ("possibly proves" is certain). A hedge in one sentence does not
soften a causal sentence next to it. Negation is recognised only where it governs the causal verb,
so "the shift caused the drop, although this does not prove anything" is still causal. The noun
phrase "root cause" on its own asserts nothing; "the root cause is X" does.

### Step 2: compare with the strength of the evidence

| Strength | What provides it |
|---|---|
| OBSERVATION | a monitor measured something: drift, a rate, a performance drop |
| ASSOCIATION | a monitor measured a relationship: feature-label association gain or reversal |
| MECHANISM | a diagnostic showed how something changed: constant-factor fit, artifact hash differs |
| INTERVENTION | the sandbox applied the fix for that cause and the targeted findings cleared |

A fact, an association or a hypothesis needs relevant OBSERVATION-grade anomaly evidence. A causal
or certain claim needs INTERVENTION-grade evidence for the same cause. A drift alert and a
performance drop together are two observations; they never amount to an intervention. So:

> "The feature distribution shift caused the model performance degradation." - **rejected**
> on a drift alert and a metric drop.
>
> "A feature distribution shift was detected alongside model performance degradation. The
> available evidence does not establish causality." - **accepted** on the same evidence.

### Evidence checks applied to every claim

| Check | Rule |
|---|---|
| Existence | a cited id must be in the investigation's evidence; others are recorded as fabricated and stripped |
| Provenance | evidence must come from a monitor or a diagnostic tool. Runbook text is reference material, not evidence |
| Relevance | evidence must concern the claim's subject: the named feature, or a check that the rule engine associates with the named cause. Performance and "what did not happen" evidence bears on every cause |
| Type and strength | as in the table above |
| Contradiction | citing as support what the rule engine counts against the same cause, or denying an anomaly the evidence records for the named feature |
| Measured effect | a statement about a performance change needs a recorded performance anomaly |

### Verdicts and what happens

| Verdict | When | Effect on a ranked cause | Effect on an explanation |
|---|---|---|---|
| supported | evidence meets the wording | kept | kept |
| rejected | unsupported, contradictory, or causal wording with less than MECHANISM evidence | unsupported or contradictory: the item is dropped. Causal wording: reasoning replaced by the rule engine's | the explanation is discarded; the deterministic statement of facts is shown |
| escalated | causal wording, a named cause, MECHANISM-grade evidence, no intervention | reasoning replaced | the sentence is replaced by a hedged one and the explanation carries a review flag |

At ranking time no validation has run, so causal wording in a ranking is never supported. An
explanation is written after sandbox validation; a passed validation for the same cause is the
only thing that supports a causal sentence there. Escalation applies only to causal (not certain)
wording about an explicitly named cause.

Whatever the verdict, the model's raw text and every claim record are stored with the
investigation (`llm_ranking_stats.raw_items`, `llm_ranking_stats.claim_records`,
`explanation.claim_validation`).

## Counters

| Counter | Meaning |
|---|---|
| `claims_evaluated` | sentences and ranking items validated |
| `causal_claims_generated` | claims the lexicon classified as causal or certain |
| `causal_claims_supported` / `_rejected` / `_escalated` | their verdicts; the three sum to generated |
| `unsupported_claims_generated` / `_rejected` | claims with no relevant anomaly evidence; every one is rejected |
| `contradictory_claims_generated` / `_rejected` | claims that lean on contradicting evidence |
| `invalid_cause_labels_rejected` | ranking items naming a cause outside the vocabulary |
| `fabricated_evidence_ids_generated` / `_stripped` | cited ids that do not exist |
| `irrelevant_evidence_ids_stripped`, `untrusted_evidence_ids_stripped` | existing ids that fail relevance or provenance |
| `causal_claims_escaped`, `unsupported_claims_escaped`, `fabricated_evidence_ids_escaped` | found by re-scanning what actually leaves the investigation |

Rates in the report, each with numerator and denominator:
`causal_claims_not_supported_rate` = (rejected + escalated) / generated;
`causal_claims_escaped_rate` = escaped / generated;
`unsupported_claims_escaped_rate`; `fabricated_evidence_ids_escaped_rate`;
`claims_with_causal_language_rate` = causal generated / claims evaluated.

A run without counters reports them as unavailable, not as zero.

## Ordering: deterministic evidence leads

After validation, rule hypotheses the model did not mention are merged back and the whole list is
ordered by evidence fit. A cause the rule engine does not support is capped at 0.45, so a model
cannot put its own guess ahead of a better-supported rule hypothesis, either by ranking it first
or by leaving the rule hypothesis out. (The second route existed until it was observed with the
fallback model; see `docs/release-readiness.md`.)

## What "escaped" can and cannot mean

The escape counters come from running the same validator over the final output. They prove the
pipeline is internally consistent: nothing the validator recognises got through. They cannot
count an over-claim the lexicon does not recognise, because such a sentence was never counted as
generated either.

That limit is measured rather than assumed. `tests/adversarial/test_claims_adversarial.py` holds a
labelled corpus:

| Corpus | Sentences | Result |
|---|---|---|
| over-claims (causal or certain) | 28 | 28 recognised |
| acceptable statements | 18 | 0 flagged |
| over-claims phrased outside the lexicon | 4 | 0 recognised, by design of the test |

The third group ("The shift hurt the model.", "Fix the units and the errors go away.") is kept as
a test so the limit is a recorded fact; if the lexicon grows to cover one, the test fails and the
sentence moves up. The corpus was written by the author of the lexicon, so 28 of 28 is a
regression guard, not an estimate of recall on unseen text.

Other things not checked: numbers quoted in prose are not compared with evidence values; the
relevance rule is a table of checks per cause, not an understanding of the sentence; only English
phrasing is covered.

## Tests

- `tests/unit/test_claims.py` (57): language classification, the strength policy, provenance,
  relevance, contradiction, counters, the final-output audit.
- `tests/adversarial/test_claims_adversarial.py` (67): the corpus; unsupported causal assertions;
  correlation presented as causation; incorrect and fabricated evidence references; contradictory
  evidence; a hypothesis presented as fact; a valid causal claim backed by a passed intervention;
  an intervention for one cause not licensing a claim about another; whole investigations in which
  an over-claiming explanation is discarded and the policy decision is unchanged.
