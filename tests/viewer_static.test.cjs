/* Optional: node --test tests/viewer_static.test.cjs
   Pure static-viewer helpers on synthetic data. No packages, DOM or server. */
"use strict";
const assert = require("node:assert/strict");
const test = require("node:test");
const path = require("node:path");
const E = require(path.join(__dirname, "../src/atif_scan/browser/viewer/evidence.js"));

const wide = "🌿漢é";
const field = (text, extra = {}) => ({ part: "result", index: 0, field: 0, label: "result 0", status: "text", text, ...extra });
const trial = steps => ({ id: "0", input_status: "available", coverage: {}, findings: [], steps });

test("offsets count code points, not UTF-16 units", () => {
  const text = wide.repeat(3) + "NEEDLE" + wide;
  assert.deepEqual(E.literalMatches(text, "NEEDLE"), [{ start: 9, end: 15 }]);
  assert.equal(E.length(field(text)), 18);
  const p = E.page(field(text), 9, 6);
  assert.equal(p.text, "NEEDLE");
  assert.deepEqual([p.previous, p.next], [3, 15]);
});

test("pages reject offsets outside the masked field", () => {
  assert.throws(() => E.page(field("abc"), 4));
  assert.throws(() => E.page(field("abc"), -1));
  assert.throws(() => E.page(field("abc"), 0.5));
});

test("highlights only proven spans, clipped to the page", () => {
  const span = { focus_status: "exact", highlight_start: 5, highlight_end: 15 };
  assert.deepEqual(E.clip({ offset: 10, end: 20 }, span), [0, 5]);
  assert.equal(E.clip({ offset: 20, end: 30 }, span), null);
  assert.equal(E.clip({ offset: 0, end: 30 }, { ...span, focus_status: "field" }), null);
  assert.equal(E.clip({ offset: 0, end: 30 }, { focus_status: "exact", highlight_start: null }), null);
  assert.equal(E.focusOffset(field("x".repeat(2000)), { ...span, highlight_start: 1000, highlight_end: 1001 }), 600);
  assert.equal(E.focusOffset(field("short"), { ...span, highlight_start: 900, highlight_end: 901 }), 0);
});

test("search is literal, spans fields and stops at 40 hits", () => {
  const many = trial([{ step: 3, role: "agent", fields: [field("a.".repeat(30)), field("a.".repeat(30), { index: 1 })] }]);
  const { hits, truncated } = E.search(many, "a.");
  assert.equal(hits.length, 40);
  assert.equal(truncated, true);
  assert.deepEqual([hits[30].index, hits[30].highlight_start], [1, 0]);
  assert.equal(E.search(many, ".*").hits.length, 0);
  assert.throws(() => E.search(many, ""));
  assert.throws(() => E.search(many, "x".repeat(151)));
});

test("neighbours follow recording order, not numeric step IDs", () => {
  const steps = [10, 20, 45, 55, 70, 90, 150].map(step => ({ step, role: "agent", fields: [field("x")] }));
  assert.deepEqual(E.context(trial(steps), 70).map(s => s.step), [45, 55, 70, 90, 150]);
  assert.deepEqual(E.context(trial(steps), 10).map(s => s.step), [10, 20, 45]);
  assert.throws(() => E.context(trial(steps), 11));
});

test("matches outrank unresolved checks; unknown is a gap, never clean", () => {
  const f = (severity, status = "match") => ({ severity, status, locations: [] });
  const found = [f("info"), f("high", "unknown"), f("medium"), f("critical")];
  assert.deepEqual(E.ordered(found).map(x => `${x.severity}/${x.status}`),
    ["critical/match", "medium/match", "info/match", "high/unknown"]);
  const unresolved = { ...trial([]), findings: [f("low", "unknown")] };
  assert.equal(E.hasGap(unresolved), true);
  assert.equal(E.needsAttention(unresolved), true);
  assert.equal(E.topPriority(unresolved), null);
  assert.equal(E.hasGap({ ...trial([]), input_status: "unavailable" }), true);
  assert.equal(E.needsAttention({ ...trial([]), findings: [f("low")] }), false);
});

test("reward unknown is not zero", () => {
  assert.equal(E.rewardLabel(null), "Reward unknown");
  assert.equal(E.rewardLabel(Number.NaN), "Reward unknown");
  assert.equal(E.rewardLabel(0), "Unrewarded");
  assert.equal(E.rewardLabel(0.5), "Rewarded");
});

test("coverage keeps gaps explicit and drops known-count bookkeeping", () => {
  const c = E.coverageLines({ incomplete: false, input_error: null,
    coverage_gaps: { behavioural: ["lookup.synthetic"], telemetry: ["integrity.synthetic"] },
    recording_gaps: { web_call_counts_trials_known: 1, trials: 1, web_calls_without_usable_result: 2, pairing_unresolved: 0 } });
  assert.equal(c.incomplete, true);
  assert.deepEqual(c.lines, ["web calls without usable result: 2"]);
  assert.deepEqual(c.telemetry, ["integrity.synthetic"]);
});

test("field labels show sizes or explicit availability", () => {
  assert.equal(E.charLabel(field(wide.repeat(1000))), "3,000 chars");
  assert.equal(E.charLabel(field("", { status: "empty_or_absent" })), "empty or absent");
  assert.equal(E.stepSummary({ fields: [field("x", { part: "message" }), field("y", { part: "call" }),
    field("z", { part: "call_info" }), field("w")] }), "message · 1 call · 1 result");
});

test("explained findings stay listed but are not leads or a trial's priority", () => {
  const canary = { severity: "medium", status: "match", expected_by: ["expected.synthetic"], locations: [] };
  const lead = { severity: "low", status: "match", expected_by: [], locations: [] };
  const t = { ...trial([]), findings: [canary, lead] };
  assert.equal(E.isLead(canary), false);
  assert.equal(E.topPriority(t), "low");
  assert.deepEqual(E.ordered([canary, lead]), [lead, canary]);
});

test("unread context says where and what wasn't inspected", () => {
  assert.equal(E.unreadText({ reason: "media", location: { step: 6, part: "result", index: 0, field: 0 } }),
    "Not inspected: step 6 result 0 contains an image or other media whose content wasn't inspected; it may have shown this.");
  assert.equal(E.unreadText({ reason: "prompt_not_recorded", location: null }),
    "Not inspected: the task prompt wasn't recorded.");
  assert.match(E.unreadText({ reason: "made_up", location: null }), /couldn't be inspected/);
});

test("missing results say why their output wasn't checked", () => {
  const at = { step: 76, part: "call_info", index: 0, field: 0 };
  assert.match(E.unreadText({ reason: "run_ended", location: at }), /^Not inspected: step 76 call 0 never returned: the run ended/);
  assert.match(E.unreadText({ reason: "result_compacted", location: at }), /context compaction/);
  assert.match(E.unreadText({ reason: "undecidable", location: { ...at, part: "call" } }), /computed at runtime/);
});

test("run strip formats amounts and never presents an estimate as a bill", () => {
  assert.equal(E.compact(550_100_000), "550.1M");
  assert.equal(E.usd(487.09256), "$487.09");
  assert.equal(E.duration(131_366), "36h 29m");
  const observed = E.costHeadline({ recorded_usd: 0, without_cost: 85, observed_usd: 487.09, observed_trials: 85 });
  assert.equal(observed.value, "$487.09");
  assert.match(observed.caption, /not a final bill/);
  assert.match(E.costHeadline({ recorded_usd: 0, estimate_usd: 12, estimate_source: "declared" }).value, /^est\. /);
  assert.equal(E.costHeadline({ recorded_usd: 0 }).value, "No cost");
  assert.deepEqual(E.harnessLine([{ agent: "fast-agent", version: "0.10.42", model: "grok-4.7", trials: 83 },
    { agent: "fast-agent", version: "0.10.42", model: null, trials: 2 }]),
    { value: "fast-agent 0.10.42", caption: "grok-4.7 (83) · model not recorded (2)" });
  assert.match(E.trialFacts({ agent_name: "fast-agent", model_name: null, llm_calls: 74, context_compactions: 1,
    error_type: "NonZeroAgentExitCodeError" }), /model not recorded · 74 LLM calls · no cost recorded · 1 context compaction · errored/);
});

test("recording checks are neither leads nor a trial's priority", () => {
  const cost = { severity: "medium", status: "match", category: "recording", expected_by: [], locations: [] };
  assert.equal(E.isLead(cost), false);
  assert.equal(E.topPriority({ ...trial([]), findings: [cost] }), null);
});

test("a prompt that names the benchmark reads as ambiguity, not a gap", () => {
  assert.equal(E.unreadText({ reason: "prompt_names_benchmark", location: { step: 1, part: "message", index: 0, field: 0 } }),
    "Can't tell apart: step 1 message already names the benchmark, so a result naming it is no signal.");
});

test("accounting checks show their own calculation", () => {
  assert.deepEqual(E.calcLines({ check_id: "integrity.tokens_exceed_recorded_calls",
    measure: { prompt_tokens: 1108874, llm_calls: 33, limit_per_call: 2000000, per_call: 33602.2 } }),
    ["1,108,874 prompt tokens ÷ 33 LLM calls = 33,602.2 per call · flagged above 2,000,000 per call"]);
  assert.equal(E.calcLines({ check_id: "integrity.cost_missing", measure: { prompt_tokens: 40103, completion_tokens: 2411, cost_usd: null } })[0],
    "Trajectory totals: 40,103 prompt + 2,411 completion tokens · cost: not recorded");
  assert.match(E.calcLines({ check_id: "integrity.output_token_ratio",
    measure: { visible_chars: 9000, output_tokens: 3000, chars_per_token: 3, min: 1, max: 8, basis: "includes_reasoning" } })[1],
    /Expected 1–8 per token · output tokens include reasoning/);
  assert.deepEqual(E.calcLines({ check_id: "x", measure: {} }), []);
  assert.equal(E.runRecordsNote({ unread: [{ reason: "usage_not_recorded" }] }, { input_tokens: 40103, output_tokens: 2411 }),
    "The run's records (not the trajectory) have 40.1k input · 2.4k output tokens.");
  assert.match(E.unreadText({ reason: "usage_not_recorded", location: null }), /^Couldn't compare: the trajectory records no token totals/);
});

test("an unknown says whether figures were missing or the trace was unread", () => {
  assert.match(E.unknownLead({ unread: [{ reason: "calls_without_usage" }] }), /figures this check needs/);
  assert.match(E.unknownLead({ unread: [{ reason: "media" }] }), /couldn't be inspected/);
  assert.match(E.unknownLead({ unread: [] }), /could not be resolved/);
});

test("trial facts carry provider retries and how the trajectory ended", () => {
  assert.match(E.trialFacts({ stream_retry_attempts: 3, termination_error: "ResponsesWebSocketError", error_type: "NonZeroAgentExitCodeError" }),
    /3 provider retries · trajectory ended: ResponsesWebSocketError · errored: NonZeroAgentExitCodeError/);
});

test("a failed trial says what happened, from its facts", () => {
  const safety = E.outcome({ reward: 0, facts: { error_type: "AgentSafetyStopError", final_stop_reason: "safety",
    safety_provider: "anthropic", safety_reason: "refusal", safety_category: "cyber", agent_steps: 1, output_tokens: 0,
    failed_phase: "after_agent_execution", setup_duration_sec: 66.8, agent_duration_sec: 7.3, verifier_duration_sec: 10 } });
  assert.equal(safety.cls, "safety");
  assert.equal(safety.headline, "Stopped by Anthropic's safety system (refusal · cyber)");
  assert.match(safety.lines[0], /first model call, before any output/);
  assert.deepEqual(safety.phases.map(p => [p.name, p.failed]), [["setup", false], ["agent", true], ["verifier", false]]);
  // a generic agent exit caused by the model service reads as a provider error
  assert.equal(E.errorClass({ error_type: "NonZeroAgentExitCodeError", termination_error: "ResponsesWebSocketError" }), "provider");
  assert.equal(E.errorClass({ error_type: "RuntimeError", failed_phase: "agent_setup" }), "infrastructure");
  assert.match(E.outcome({ facts: { error_type: "SandboxError", failed_phase: "environment_setup" } }).lines[0], /never ran/);
  assert.equal(E.outcome({ facts: {} }), null);
});

test("replaced trials are evidence, and both ends of a replacement say so", () => {
  const original = { selection: { role: "replaced", id: "synthetic-run-rep-001", replaced_trial: "synthetic-task__Orig001",
    replaced_error: "SandboxError", failure_phase: "agent-or-setup", replaced_by: "synthetic-task__Repl001" } };
  const replacement = { selection: { role: "canonical", id: "synthetic-run-rep-001", replaced_trial: "synthetic-task__Orig001",
    replaced_error: "SandboxError", failure_phase: "agent-or-setup" } };
  assert.equal(E.counted(original), false);
  assert.equal(E.counted(replacement), true);
  assert.equal(E.counted({}), true);
  const a = E.lineageNote(original);
  assert.equal(a.headline, "Replaced: not counted");
  assert.match(a.text, /failed \(SandboxError, agent-or-setup\) and was re-run as synthetic-task__Repl001/);
  assert.equal(a.target, "synthetic-task__Repl001");
  const b = E.lineageNote(replacement);
  assert.equal(b.headline, "Replacement for synthetic-task__Orig001");
  assert.equal(b.target, "synthetic-task__Orig001");
  assert.match(E.lineageNote({ selection: { role: "replaced", replaced_by: null } }).text, /hasn't run yet/);
  assert.equal(E.lineageNote({ selection: { role: "canonical" } }), null);
});
