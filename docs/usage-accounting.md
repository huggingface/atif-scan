# Scoped usage comparisons

Canonical trajectory and run token counts are never rewritten to repair a comparison.
Usage comparisons are not provider-billing verification.

## Embedded fast-agent child usage

Fast-agent can include summarisation calls in embedded `subagent_trajectories` and
in the parent final totals, while the parent's steps contain only root-agent activity.

The loader derives root-only usage only when the explicit `root_*_tokens` and
`subagent_*_tokens` fields reconcile with both canonical totals and the embedded
children's final totals for input, output and cached input. Children must have unique
nonempty trajectory IDs, recorded steps, usable totals, and no explicit incomplete-call
flag. Missing, malformed, duplicate or inconsistent records do not explain away a gap.
No child commands, URLs or text are executed or promoted into scanned root history.

Step comparisons and authored-text ratios use those reconciled root totals.
Reported canonical tokens, costs and provider-usage completeness remain unchanged.
Root reasoning tokens can be derived only when every child has a reasoning total.

## Per-step reasoning fallback

Without final completion totals, text/token comparisons use only authored steps that
record completion tokens. The loader accepts `metrics.extra.reasoning_tokens` only
as a non-negative integer no larger than that step's completion count.

If every compared step has a valid split (including explicit zero), the ratio excludes
both reasoning text and reasoning tokens. Missing or invalid splits remain unknown;
a partial split is not subtracted. Step reasoning counts cannot repair an unknown split
in final totals covering a different scope.

A plausible ratio for metered output does not clear attempts without usage. Partial
usage, retry counts and explicit provider-usage incompleteness remain evidence gaps.
The ratio is a character-count sanity check, not a tokenizer measurement.

## Totals that cover only the last call

Some harnesses write one model call's usage as the run's `final_metrics` totals when a
run is cut off (indusagi 0.2.12/0.2.13 timeouts and crashes). The totals then equal the
last metered step's prompt and completion tokens exactly, while two or more metered
steps sum to more; Harbor's recorded tokens and cost copy the short totals.

`integrity.totals_are_last_call` reports the pattern with the totals, the step sums and
the last step's figures. Canonical totals are not rewritten: the brief adds what the
steps record beyond them (step sums minus totals) as tokens under TOKENS and, priced at
the run's rates, as an estimate outside the recorded total under COST. The text/token
ratio uses the steps' own tokens for these trials, since the whole run's text can't be
compared with one call's output.

Step metrics where every step's prompt and completion tokens are at least the previous
step's may be running totals, for which totals equal to the last step are correct: the
check is unknown (`step_metrics_may_be_cumulative`) and the totals are used as recorded.
