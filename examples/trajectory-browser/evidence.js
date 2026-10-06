/* Pure evidence-navigation helpers. Offsets count Unicode code points, as Python does. */
(function (root) {
  "use strict";
  const PAGE_SIZE = 1800;
  function locate(trial, ref) {
    const step = trial.steps.find(s => s.id === ref.step);
    if (!step) throw new Error("Unknown step");
    const field = step.fields.find(f => f.part === ref.part &&
      f.index === (ref.index ?? 0) && f.field === (ref.field ?? 0));
    if (!field) throw new Error("Unknown field");
    return { step, field };
  }
  function page(field, offset, limit = PAGE_SIZE) {
    const chars = Array.from(field.text ?? "");
    if (!Number.isSafeInteger(offset) || offset < 0 || offset > chars.length)
      throw new Error("Invalid masked offset");
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 6000)
      throw new Error("Invalid page size");
    const end = Math.min(offset + limit, chars.length);
    return { text: chars.slice(offset, end).join(""), offset, end, total: chars.length,
      previous: offset > 0 ? Math.max(0, offset - limit) : null,
      next: end < chars.length ? end : null, status: field.status };
  }
  function focusOffset(field, ref) {
    const total = Array.from(field.text ?? "").length;
    return Number.isSafeInteger(ref.start) && ref.start >= 0 && ref.start < total
      ? Math.max(0, ref.start - 550) : 0;
  }
  function context(trial, stepId, radius = 2) {
    const index = trial.steps.findIndex(s => s.id === stepId);
    if (index < 0) throw new Error("Unknown step");
    return trial.steps.slice(Math.max(0, index - radius), index + radius + 1);
  }
  function literalMatches(text, term, maxHits = 40) {
    const chars = Array.from(text), needle = Array.from(term);
    if (!needle.length || needle.length > 150) throw new Error("Use 1–150 characters");
    const hits = [];
    for (let i = 0; i <= chars.length - needle.length && hits.length < maxHits; i++) {
      if (chars.slice(i, i + needle.length).join("") === term) {
        hits.push({ start: i, end: i + needle.length });
        i += needle.length - 1;
      }
    }
    return hits;
  }
  function search(trial, term) {
    const hits = [];
    for (const step of trial.steps) {
      for (const field of step.fields) {
        for (const hit of literalMatches(field.text ?? "", term, 40 - hits.length)) {
          hits.push({ step: step.id, part: field.part, index: field.index, field: field.field, ...hit });
        }
        if (hits.length >= 40) return hits;
      }
    }
    return hits;
  }
  const api = { PAGE_SIZE, locate, page, focusOffset, context, literalMatches, search };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Evidence = api;
})(globalThis);
