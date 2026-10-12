# Scoped usage comparisons

Canonical trajectory and run token counts are never rewritten to repair a comparison.
Usage comparisons are not provider-billing verification.

## OpenHands / ALE export boundaries

Treat the ATIF export and the native OpenHands recording as different evidence scopes.
An ATIF `reasoning_content` field establishes only that some text was exported there;
its presence does not establish that the native event's provider `reasoning_content`
was retained, or that it is complete. A `think` tool's `thought` argument is explicit
model-authored tool prose, not proof of exposed provider-internal reasoning.

ALE-style exports can omit per-step usage and the final reasoning-token split even
when OpenHands' native `base_state.json` records `stats.usage_to_metrics` with
`token_usages` and `accumulated_token_usage.reasoning_tokens`. Missing reasoning
counts remain unknown, not zero. The output-token ratio cannot check its lower bound
without a valid split; short exported reasoning is not evidence of few reasoning tokens.
Native transcript `reasoning_content`, `thought`, `thinking_blocks` and
`responses_reasoning_item` are distinct fields and must not be conflated. In particular,
an export may put the native action's `summary` in ATIF `reasoning_content` while omitting
the provider reasoning. That summary is not a replacement for the native reasoning.

Native reasoning text alongside a reported zero reasoning-token count makes the split
questionable: do not compare visible text alone against all completion tokens and call
the difference lost output. A combined character count is a descriptive sanity check,
not a verified split. Conversely, reasoning tokens exceeding completion tokens cannot
be subtracted to produce a valid visible-output denominator.

Native `Condensation` events can carry separate summarisation responses and usage.
Reconcile those responses separately before comparing totals with an export that omits
them; a condensation event alone does not prove earlier events are absent from the
saved transcript. Rejected native tool calls may retain original arguments even when
the typed action export replaces them with `{}`. Recovering those arguments must preserve
the rejection status, not relabel the attempted action as executed.

Also verify the prompt/cache convention before using a scorecard or price estimate.
The scanner expects prompt totals to include cached input. An exporter that writes
uncached input as `total_prompt_tokens` is not using that convention: for example,
100 uncached plus 900 cached tokens means 1,000 prompt tokens, not 100. Do not infer a
repair merely from the agent name or from cached tokens exceeding prompt tokens.
Reconciliation needs the matching native usage scope. All-zero totals alongside
recorded model activity do not establish zero consumption or a free run.

The standard ATIF loader does not automatically read native companion logs or saved
state, nor silently replace canonical totals with their contents. Those files can
contain credentials. A future companion adapter must explicitly bind the recordings,
allowlist usage fields, preserve missing/partial usage, and keep recovered provider
reasoning distinct from tool prose and exporter summaries.

## Pi / ALE export boundaries

Pi's native JSONL can carry provider text and per-response usage together. Use completed
assistant `message_end` records for that scope: streaming updates, `message_start` and
`turn_end` are not independent billable responses. Track errors and unfinished streams
separately; recorded zero usage does not establish that an attempted request was free.

The ALE Pi parser at revision `d9abc0734b56ea34116c5bfcbdd0b808269ab9e2` looks for
`text` inside thinking blocks, whereas Pi can record `{type: "thinking", thinking: ...}`.
It also omits the native `usage.reasoning` split and aggregates message usage instead
of attaching it to each response. Consequently, absent exported reasoning need not
mean the native recording withheld it. This inspected revision is not established as
the producing revision of a particular downloaded run.

Nested tool-result `content` can be serialized and cut to a 2,000-character prefix in
an observation, while the complete tool-result text is separately exported as a system
message. That preserves text somewhere, but not its observation attribution. Neither
copy is agent-authored output; system text must not repair an authored-output ratio.
Images nested in native results may also be absent from exported media.

Pi `compaction_end.result` can carry a summary and separate usage. Matching final totals
to assistant-message usage alone does not account for that summarisation scope. Check
it separately rather than silently adding it to canonical counts. Likewise, when native
`totalTokens = input + output + cacheRead + cacheWrite`, `input` is not an inclusive
prompt total. Preserve the recorded fields and make any reconciled comparison explicit.

## Qwen Code and ZCode / ALE export boundaries

Preserved reasoning text does not establish accounting fidelity. A parser can retain
assistant messages' usage, then append a cumulative run-summary usage as another
metered step. Summing both scopes double-counts their overlap. Match summary fields
against per-response records before adding anything; unexplained excess in a summary
is not automatically a subagent or compaction charge.

Conversely, a native-to-ALE conversion can preserve per-request usage on system steps,
while a subsequent ATIF conversion drops those metrics but keeps the final totals.
Matching aggregate totals then says nothing about recoverable per-response attribution.
ZCode request totals can include main, subagent, web-processing and compaction requests;
a final summary covering only main requests is not a like-for-like comparator.
Compaction summaries, failed requests, retries and tool-result status flags must be
checked separately from successful streamed messages and ordinary tool-result strings.

## Direct ATIF consistency checks versus companion fidelity

`integrity.cached_tokens_exceed_prompt` flags a final cached-input count larger than
its inclusive prompt count. It does not infer or apply a corrected accounting convention.
Missing or invalid operands are unknown, and step sums are not substituted.

`integrity.step_count_mismatch` compares `final_metrics.total_steps` against the root
trajectory's recorded steps. Folding environment steps into observations can leave an
exporter's old count behind. The schema permits documented scope differences in root
`notes`: when nonempty notes accompany a mismatch, the check stays unknown rather than
interpreting prose or accepting it as proof. Children are not silently added to the root.

These are recording-review findings, not misconduct verdicts. Missing optional per-step
usage or reasoning is an evidence limitation, not by itself a proven export defect.
The built-in ATIF checks do not currently perform native-companion fidelity comparisons:
proving that reasoning, summaries, usage or outcomes were lost requires the matching
native evidence. A valid schema, plausible token ratio or no matched consistency check
must not be presented as certification of a faithful export.

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
