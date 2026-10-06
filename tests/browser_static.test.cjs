/* Optional: node --test tests/browser_static.test.cjs
   Synthetic data and an in-memory DOM/transport only. No packages or server. */
"use strict";
const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const asset = path.join(__dirname, "../src/atif_scan/browser/static");
const H = require(path.join(asset, "browser.js"));
const defaults = { trial: "", check: "", priority: "medium+", reward: "all", status: "match", review: "all" };
const finding = (id = "0", status = "match") => ({
  id, check_id: "synthetic.credentials", title: "Synthetic check", severity: "medium", status,
  locations: [{ step: 7, part: "message", index: 0, field: 0 }],
  expected_by: [], feedback: { verdict: "unreviewed", note: "" }
});
const trial = id => ({
  id, task: `synthetic-${id}`, input_id: `synthetic-${id}`, reward: null, input_status: "available",
  findings: [finding(), finding("1", "unknown")],
  coverage: { incomplete: true, coverage_gaps: ["synthetic_gap"] },
  steps: [7, 20, 400].map(step => ({ step, role: "agent", fields: [{ part: "message", index: 0, field: 0 }] }))
});
const page = (text = "synthetic", offset = 0) => ({
  text, offset, end_offset: offset + Array.from(text).length, total_length: 9000, next_offset: offset + Array.from(text).length,
  limit: 3000, status: "text", understood: true, media: false
});

test("default filter prioritizes matches medium+; unknown never becomes match via feedback", () => {
  const f = finding();
  assert.equal(H.findingMatches(f, defaults), true);
  assert.equal(H.findingMatches({ ...f, severity: "low" }, defaults), false);
  f.status = "unknown";
  f.feedback.verdict = "false_positive";
  assert.equal(H.findingMatches(f, defaults), false);
  assert.equal(H.findingMatches(f, { ...defaults, status: "gaps", review: "false_positive" }), true);
  assert.equal(H.findingMatches(f, { ...defaults, status: "all", check: "CREDENTIALS" }), true);
  assert.equal(H.findingMatches(f, { ...defaults, status: "all", check: "absent" }), false);
});
test("reward and empty trials preserve unknowns", () => {
  for (const value of [null, undefined, "1", NaN]) {
    assert.equal(H.rewardMatches(value, "unknown"), true);
    assert.equal(H.rewardMatches(value, "zero"), false);
  }
  assert.equal(H.rewardMatches(0, "zero"), true);
  assert.equal(H.rewardMatches(-1, "negative"), true);
  assert.equal(H.rewardMatches(1, "positive"), true);
  const t = { ...trial("0"), findings: [] };
  assert.equal(H.trialMatches(t, defaults), false);
  assert.equal(H.trialMatches(t, { ...defaults, status: "all", priority: "all" }), true);
});
test("recorded neighbours and Unicode paging do not use numeric IDs or UTF-16 length", () => {
  const steps = [1, 7, 20, 50, 400, 999].map(step => ({ step }));
  assert.deepEqual(H.neighbours(steps, 50).map(s => s.step), [7, 20, 50, 400, 999]);
  assert.deepEqual(H.neighbours(steps, 888), []);
  assert.equal(H.neighbours(steps, 1, true).length, 6);
  assert.equal(H.charCount("a😀b"), 3);
  assert.equal(H.previousOffset({ offset: 3001, limit: 3000 }), 1);
  assert.equal(H.previousOffset({ offset: 2, limit: 3000 }), 0);
  assert.equal(H.dirty({ verdict: "unclear", note: "a" }, { verdict: "unclear", note: "b" }), true);
});

// A deliberately small DOM: textContent is literal and HTML interpretation fails.
class Element {
  constructor(tag = "") { this.tagName = tag; this.children = []; this.events = {}; this.value = ""; this.textContent = ""; this.disabled = false; this.hidden = false; this.attrs = {}; }
  set textContent(value) { this.children = []; this.text = String(value); }
  get textContent() { return this.text + this.children.map(n => n.textContent).join(""); }
  set innerHTML(_) { throw Error("HTML interpretation forbidden"); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.text = ""; this.children = nodes; }
  setAttribute(k, v) { this.attrs[k] = v; }
  addEventListener(k, fn) { this.events[k] = fn; }
  focus() {}
  fire(type = "click") { return this.events[type]?.({ preventDefault() {} }); }
}
function harness(hash = "#token=synthetic-secret&trial=0") {
  const elements = {};
  const html = fs.readFileSync(path.join(asset, "index.html"), "utf8");
  for (const match of html.matchAll(/id="([^"]+)"/g)) elements[match[1]] = new Element();
  for (const [key, value] of Object.entries({ priority: "medium+", reward: "all", status: "match", review: "all" })) elements[key].value = value;
  const calls = [], confirmations = [];
  const window = { confirm: () => confirmations.shift() ?? true, addEventListener() {} };
  const context = {
    document: { getElementById: id => elements[id], createElement: tag => new Element(tag), createTextNode: text => { const n = new Element("#text"); n.textContent = text; return n; } },
    window, location: { hash }, URLSearchParams,
    fetch: (url, options) => new Promise((resolve, reject) => calls.push({
      url, options, body: JSON.parse(options.body),
      resolve: data => resolve({ ok: true, json: async () => structuredClone(data) }),
      fail: () => reject(Error("sensitive error must never be displayed"))
    }))
  };
  vm.runInNewContext(fs.readFileSync(path.join(asset, "browser.js"), "utf8"), context);
  return { elements, calls, confirmations, context };
}
const tick = () => new Promise(resolve => setImmediate(resolve));
async function loaded() {
  const h = harness();
  h.calls[0].resolve({ trials: [trial("0"), trial("1")] });
  await tick();
  h.elements["trial-list"].children[0].fire();
  h.calls[1].resolve(trial("0"));
  await tick();
  return h;
}

test("fixed POST transport, hash-only token, and missing token has no requests", async () => {
  const h = harness();
  const call = h.calls[0];
  assert.equal(call.url, "/api/overview");
  assert.equal(call.options.method, "POST");
  assert.equal(call.options.headers.Authorization, "Bearer synthetic-secret");
  assert.equal(call.options.headers["Content-Type"], "application/json");
  assert.equal(call.options.redirect, "error");
  assert.equal(call.options.mode, "same-origin");
  assert.equal(h.context.location.hash, "#token=synthetic-secret&trial=0");
  call.fail();
  await tick();
  assert.doesNotMatch(h.elements.connection.textContent, /sensitive error|synthetic-secret/);
  assert.equal(harness("").calls.length, 0);
});

test("late trial response cannot overwrite newer selection", async () => {
  const h = harness(), e = h.elements;
  h.calls[0].resolve({ trials: [trial("0"), trial("1")] });
  await tick();
  e["trial-list"].children[0].fire();
  e["trial-list"].children[1].fire();
  h.calls[2].resolve(trial("1"));
  await tick();
  h.calls[1].resolve(trial("0"));
  await tick();
  assert.match(e["trial-title"].textContent, /synthetic-1/);
  assert.equal(h.calls.at(-1).body.trial, "1");
});

test("field races, failure clearing, Unicode server offsets and literal rendering", async () => {
  const h = await loaded(), e = h.elements;
  const old = h.calls[2];
  e.later.fire();
  const current = h.calls[3];
  current.resolve(page("<script>synthetic</script>😀", 3001));
  await tick();
  old.resolve(page("STALE"));
  await tick();
  assert.equal(e["field-text"].textContent, "<script>synthetic</script>😀");
  e["previous-page"].fire();
  assert.equal(h.calls.at(-1).body.offset, 1);
  assert.equal(e["field-text"].textContent, "");
  assert.equal(e["next-page"].disabled, true);
  h.calls.at(-1).fail();
  await tick();
  assert.equal(e["field-text"].textContent, "");
  assert.equal(e["go-offset"].disabled, true);
  assert.doesNotMatch(e["field-state"].textContent, /sensitive error/);
});

test("finding switch invalidates both field and search responses", async () => {
  const h = await loaded(), e = h.elements;
  const oldField = h.calls[2];
  e["search-query"].value = "synthetic";
  e["search-form"].fire("submit");
  const oldSearch = h.calls.at(-1);
  e.status.value = "all";
  e.status.fire("input");
  e["finding-list"].children[1].fire();
  oldField.resolve(page("STALE FIELD"));
  oldSearch.resolve({ matches: [{ step: 7, part: "message", index: 0, field: 0, offset: 0 }], truncated: true });
  await tick();
  assert.equal(e["field-text"].textContent, "");
  assert.equal(e["search-results"].children.length, 0);
  assert.match(e["finding-meta"].textContent, /Finding 1/);
  h.calls.at(-1).resolve(page("NEW FIELD"));
  await tick();
  assert.equal(e["field-text"].textContent, "NEW FIELD");
});

test("search edits invalidate pending matches; capped search navigates by server Unicode offset", async () => {
  const h = await loaded(), e = h.elements;
  e["search-query"].value = "😀";
  e["search-form"].fire("submit");
  h.calls.at(-1).resolve({ matches: [{ step: 20, part: "message", index: 0, field: 0, offset: 8 }], truncated: true });
  await tick();
  assert.match(e["search-state"].textContent, /cap reached/);
  e["search-results"].children[0].fire();
  assert.equal(h.calls.at(-1).body.offset, 8);
  e["search-form"].fire("submit");
  const pending = h.calls.at(-1);
  e["search-query"].value = "different";
  e["search-query"].fire("input");
  pending.resolve({ matches: [{ step: 20 }], truncated: false });
  await tick();
  assert.equal(e["search-results"].children.length, 0);
});

test("unsaved feedback can cancel switching; explicit save is bound and does not clear gaps", async () => {
  const h = await loaded(), e = h.elements;
  const coverage = e.coverage.textContent;
  e.note.value = "Synthetic note";
  e.note.fire("input");
  assert.equal(h.calls.filter(c => c.url === "/api/feedback").length, 0);
  h.confirmations.push(false);
  e["trial-list"].children[1].fire();
  assert.match(e["trial-title"].textContent, /synthetic-0/);
  assert.equal(e.note.value, "Synthetic note");
  e.verdict.value = "false_positive";
  e.verdict.fire("change");
  e["feedback-form"].fire("submit");
  const save = h.calls.at(-1);
  assert.deepEqual(save.body, { trial: "0", finding: "0", verdict: "false_positive", note: "Synthetic note" });
  e["trial-list"].children[1].fire();
  assert.match(e["trial-title"].textContent, /synthetic-0/);
  save.resolve({ verdict: "false_positive", note: "Synthetic note" });
  await tick();
  assert.match(e["feedback-state"].textContent, /saved/);
  assert.equal(e.coverage.textContent, coverage);
  assert.match(e["finding-meta"].textContent, /status: match/);
  assert.equal(e.save.disabled, true);
  e["trial-list"].children[1].fire();
  assert.equal(e.note.value, "");
  assert.equal(h.calls.at(-1).body.trial, "1");
});

test("save failures keep edits; invalid locations never leave old fields enabled", async () => {
  const h = await loaded(), e = h.elements;
  h.calls[2].resolve(page());
  await tick();
  e.note.value = "Keep this draft";
  e.note.fire("input");
  e["feedback-form"].fire("submit");
  h.calls.at(-1).fail();
  await tick();
  assert.equal(e.note.value, "Keep this draft");
  assert.equal(e.save.disabled, false);
  e["trial-list"].children[1].fire();
  const next = trial("1");
  next.findings[0].locations[0].step = 999;
  h.calls.at(-1).resolve(next);
  await tick();
  assert.equal(e["field-text"].textContent, "");
  assert.equal(e["go-offset"].disabled, true);
  assert.match(e["field-state"].textContent, /unavailable/);
});

test("run-level sync gaps are visible independently of findings and filters", async () => {
  const h = harness();
  h.calls[0].resolve({ trials: [trial("0")], report: { coverage: { sync_failed_files: 2 } } });
  await tick();
  assert.match(h.elements.connection.textContent, /2 sync files unavailable/);
  assert.match(h.elements["run-coverage"].children[0].textContent, /sync_failed_files/);
});

test("web gap reason links to its call and associated result, not the first step", async () => {
  const h = harness(), e = h.elements;
  e.priority.value = "all";
  const t = trial("0"), f = finding("0");
  const call = { step: 20, part: "call", index: 2, field: 1 };
  const result = { step: 20, part: "result", index: 3, field: 0 };
  f.check_id = "integrity.web_results_not_recorded";
  f.severity = "low";
  f.locations = [{ step: 20, part: "call_info", index: 2, field: 0 }];
  f.web_gaps = [{
    step: 20, call: 2, reason: "status_only", call_location: call,
    result_location: result, pairing_reconstructed: true
  }];
  t.findings = [f];
  t.steps[1].fields = [
    { part: "call_info", index: 2, field: 0 },
    { part: "call", index: 2, field: 1 },
    { part: "result", index: 3, field: 0 }
  ];
  h.calls[0].resolve({ trials: [t] });
  await tick();
  e["trial-list"].children[0].fire();
  h.calls[1].resolve(t);
  await tick();
  assert.deepEqual(JSON.parse(JSON.stringify(h.calls[2].body)), { trial: "0", ...call, offset: 0 });
  const row = e.locations.children[0];
  assert.match(row.children[0].textContent, /Step 20 · tool call 2 — Status only/);
  assert.equal(row.children[1].textContent, "Go to tool call");
  assert.equal(row.children[2].textContent, "Inspect associated result");
  assert.match(row.children[3].textContent, /pairing reconstructed/);
  row.children[2].fire();
  assert.deepEqual(JSON.parse(JSON.stringify(h.calls[3].body)), { trial: "0", ...result, offset: 0 });
  h.calls[3].resolve(page("completed"));
  await tick();
  assert.match(e["field-title"].textContent, /Step 20 · result · index 3/);
});

test("missing result has no fictitious result button and supports call metadata", async () => {
  const h = harness(), e = h.elements;
  e.priority.value = "all";
  const t = trial("0"), f = finding("0");
  const call = { step: 7, part: "call_info", index: 0, field: 0 };
  f.locations = [call];
  f.web_gaps = [{ step: 7, call: 0, reason: "no_linked_result", call_location: call, result_location: null }];
  t.findings = [f];
  t.steps[0].fields.push({ part: "call_info", index: 0, field: 0 });
  h.calls[0].resolve({ trials: [t] });
  await tick();
  e["trial-list"].children[0].fire();
  h.calls[1].resolve(t);
  await tick();
  const row = e.locations.children[0];
  assert.equal(row.children.length, 2);
  assert.match(row.children[0].textContent, /No linked result/);
  assert.match(e["field-title"].textContent, /tool call 0 · metadata/);
  h.calls[2].resolve({ ...page("Counts only."), metadata_only: true });
  await tick();
  assert.match(e["field-state"].textContent, /missing content has not been recovered/);
});

test("actual titles and server priorities are primary; allowance metadata is readable", async () => {
  const h = harness(), e = h.elements, t = trial("0");
  const f = t.findings[0];
  f.title = "Possible credential-like value";
  f.check_id = "observation.credentials_exposed";
  f.check_version = "4";
  f.severity = "low";
  e.priority.value = "all";
  h.calls[0].resolve({ trials: [t] });
  await tick();
  e["trial-list"].children[0].fire();
  h.calls[1].resolve(t);
  await tick();
  assert.equal(e["finding-list"].children[0].children[0].textContent, f.title);
  assert.equal(e["finding-list"].children[0].children[1].textContent, `Check: ${f.check_id}`);
  assert.equal(e["review-heading"].textContent, f.title);
  assert.match(e["finding-meta"].textContent, /Priority: low/);
  assert.match(e["finding-meta"].textContent, /version 4/);
  assert.match(e["finding-meta"].textContent, /No allowance matched/);
  assert.doesNotMatch(e["finding-meta"].textContent, /\[\]|medium/);
  assert.equal(H.allowanceText(["synthetic-rule"]), "Allowance matched: synthetic-rule");
  assert.match(fs.readFileSync(path.join(asset, "index.html"), "utf8"), /No allowance matched does not establish wrongdoing/);
});

const focusedPage = (text, start, end, status = "exact", offset = 0) => ({
  ...page(text, offset), focus_status: status, highlight_start: start, highlight_end: end
});
const marks = e => e["field-text"].children.filter(n => n.tagName === "mark");

test("focus request uses location index, never raw offsets; exact XSS-like text stays literal", async () => {
  const h = await loaded(), e = h.elements;
  assert.equal(h.calls[2].url, "/api/focus");
  assert.deepEqual(h.calls[2].body, { trial: "0", finding: "0", location: 0 });
  const literal = "<script>synthetic</script>";
  h.calls[2].resolve(focusedPage(`😀${literal} tail`, 1, 1 + Array.from(literal).length));
  await tick();
  assert.equal(marks(e)[0].textContent, literal);
  assert.equal(e["field-text"].children[0].tagName, "#text");
  assert.equal(e["field-text"].textContent, `😀${literal} tail`);
  assert.match(e["focus-note"].textContent, /Exact evidence span/);
  assert.doesNotMatch(e.locations.children[0].textContent, /fallback/);
  e.locations.children[0].fire();
  assert.equal(h.calls.at(-1).url, "/api/focus");
  assert.equal(h.calls.at(-1).body.location, 0);
  assert.equal(marks(e).length, 0);
});

test("expanded masked region and field-only fallback have distinct explanations", async () => {
  const h = await loaded(), e = h.elements;
  // Only the already-masked baseline ever enters this synthetic browser fixture.
  h.calls[2].resolve(focusedPage("prefix [REDACTED] suffix", 7, 17, "masked_region"));
  await tick();
  assert.equal(marks(e)[0].textContent, "[REDACTED]");
  assert.match(e["focus-note"].textContent, /Expanded masked region, not the exact trigger/);
  e.locations.children[0].fire();
  h.calls.at(-1).resolve(focusedPage("context only", null, null, "field"));
  await tick();
  assert.equal(marks(e).length, 0);
  assert.match(e["focus-note"].textContent, /Field-level fallback: no exact span/);
});

test("Unicode global offsets clip by page, select repeated occurrence, and survive repeated paging", async () => {
  const h = await loaded(), e = h.elements;
  h.calls[2].resolve(focusedPage("😀ab😀ab", 3, 6));
  await tick();
  assert.equal(marks(e)[0].textContent, "😀ab");
  assert.equal(e["field-text"].children[0].textContent, "😀ab");
  for (const [offset, text, expected] of [[4, "ab tail", "ab"], [6, " tail", null], [0, "😀ab😀ab", "😀ab"]]) {
    e.offset.value = String(offset);
    e["offset-form"].fire("submit");
    assert.equal(h.calls.at(-1).url, "/api/segment");
    h.calls.at(-1).resolve(page(text, offset));
    await tick();
    assert.equal(marks(e)[0]?.textContent ?? null, expected);
    assert.match(e["focus-note"].textContent, expected ? /portion on this page/ : /outside this page/);
  }
  e.later.fire();
  h.calls.at(-1).resolve(page("😀ab😀ab"));
  await tick();
  assert.equal(marks(e).length, 0);
  assert.match(e["focus-note"].textContent, /Context only/);
});

test("search navigation clears focus even in the same field", async () => {
  const h = await loaded(), e = h.elements;
  h.calls[2].resolve(focusedPage("synthetic", 0, 3));
  await tick();
  e["search-query"].value = "synthetic";
  e["search-form"].fire("submit");
  h.calls.at(-1).resolve({ matches: [{ step: 7, part: "message", index: 0, field: 0, offset: 0 }], truncated: false });
  await tick();
  e["search-results"].children[0].fire();
  h.calls.at(-1).resolve(page());
  await tick();
  assert.equal(marks(e).length, 0);
  assert.match(e["focus-note"].textContent, /Context only/);
});

test("stale focus cannot overwrite a new trial or its focused selection", async () => {
  const h = await loaded(), e = h.elements, old = h.calls[2];
  e["trial-list"].children[1].fire();
  h.calls.at(-1).resolve(trial("1"));
  await tick();
  const current = h.calls.at(-1);
  assert.deepEqual(current.body, { trial: "1", finding: "0", location: 0 });
  current.resolve(focusedPage("NEW", 0, 3));
  await tick();
  old.resolve(focusedPage("STALE", 0, 5, "masked_region"));
  await tick();
  assert.equal(e["field-text"].textContent, "NEW");
  assert.equal(marks(e)[0].textContent, "NEW");
  assert.match(e["focus-note"].textContent, /Exact evidence span/);
});

test("highlight helper refuses raw or invalid bounds and clips Unicode literally", () => {
  assert.equal(H.clippedHighlight(page("abc"), { span: [0, 2] }), null);
  assert.equal(H.clippedHighlight(page("abc"), { focus_status: "field", highlight_start: 0, highlight_end: 2 }), null);
  assert.equal(H.clippedHighlight(page("abc"), { focus_status: "exact", highlight_start: null, highlight_end: 2 }), null);
  assert.deepEqual(H.clippedHighlight(page("😀ab", 10), { focus_status: "exact", highlight_start: 9, highlight_end: 12 }), [0, 2]);
});

test("new evidence location invalidates an older focus request within the same finding", async () => {
  const h = harness(), e = h.elements, t = trial("0");
  t.findings[0].locations.push({ step: 20, part: "message", index: 0, field: 0 });
  h.calls[0].resolve({ trials: [t] });
  await tick();
  e["trial-list"].children[0].fire();
  h.calls[1].resolve(t);
  await tick();
  const old = h.calls[2];
  e.locations.children[1].fire();
  const current = h.calls.at(-1);
  assert.deepEqual(current.body, { trial: "0", finding: "0", location: 1 });
  current.resolve(focusedPage("current", 0, 7));
  await tick();
  old.fail();
  await tick();
  assert.equal(e["field-text"].textContent, "current");
  assert.match(e["field-title"].textContent, /Step 20/);
  assert.equal(marks(e)[0].textContent, "current");
});
