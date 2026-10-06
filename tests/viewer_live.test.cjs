/*
 * Optional, synthetic-only live test of the static viewer (no downloads or model calls):
 * NODE_PATH=/path/to/node_modules node --test tests/viewer_live.test.cjs
 * Needs an installed Playwright and Chromium; BROWSER_EXECUTABLE may select one. Skips otherwise.
 */
"use strict";
const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const { execFileSync } = require("node:child_process");

const root = path.resolve(__dirname, "..");
const credential = "sk-proj-syntheticCredential8274abcdefghijkl";
const injection = '<img src=x onerror="window.syntheticInjected=true">';

function playwright() {
  try { return require("playwright"); } catch (error) {
    if (error.code !== "MODULE_NOT_FOUND") throw error;
    return null;
  }
}

async function exportViewer(dir) {
  const trace = path.join(dir, "synthetic.json");
  const long = Array.from({ length: 400 }, (_, i) => `synthetic line ${i} 🌿漢é\n`).join("");
  await fs.writeFile(trace, JSON.stringify({ schema_version: "ATIF-v1.7", steps: [
    { step_id: 1, source: "user", message: "Synthetic task." },
    { step_id: 2, source: "agent", message: long + `OPENAI_API_KEY=${credential} ${injection}`,
      tool_calls: [{ tool_call_id: "c1", function_name: "bash", arguments: { command: "cat /tests/test_outputs.py" } }],
      observation: { results: [{ source_call_id: "c1", content: "synthetic result" }] } },
    { step_id: 3, source: "agent", message: "Done." },
  ] }));
  const out = path.join(dir, "viewer");
  try {
    execFileSync("uv", ["run", "atif-scan", trace, "--viewer", out], { cwd: root, stdio: "pipe" });
  } catch (error) { if (error.status !== 1) throw error; }
  return out;
}

test("static viewer highlights evidence, restores links and never executes trace text", async t => {
  const pw = playwright();
  if (!pw) { t.skip("Playwright is not installed"); return; }
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "atif-viewer-"));
  const browser = await pw.chromium.launch(process.env.BROWSER_EXECUTABLE ? { executablePath: process.env.BROWSER_EXECUTABLE } : {});
  try {
    const out = await exportViewer(dir);
    const url = "file://" + path.join(out, "index.html");
    const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
    const errors = [];
    page.on("pageerror", e => errors.push(String(e)));
    page.on("console", m => { if (m.type() === "error") errors.push(m.text()); });
    await page.goto(url);
    await page.locator("#field-text mark").first().waitFor();
    assert.match(await page.locator("#field-text mark").first().textContent(), /^\/tests/);
    const hash = await page.evaluate(() => location.hash);
    assert.match(hash, /finding=\d+&loc=\d+/);

    await page.locator("summary", { hasText: "Lower priority" }).click();
    await page.locator(".finding-select", { hasText: "credential" }).click();
    const marked = await page.locator("#field-text mark").textContent();
    assert.ok(marked.includes("***") && !marked.includes(credential));
    assert.ok((await page.locator("#field-text").textContent()).includes(injection));
    assert.equal(await page.evaluate(() => window.syntheticInjected ?? null), null);

    await page.locator("[data-tab=search]").click();
    await page.fill("#evidence-query", "synthetic line 399");
    await page.locator("#search-form button").click();
    await page.locator(".search-hit").first().click();
    assert.equal(await page.locator("#field-text mark").textContent(), "synthetic line 399");

    const restored = await browser.newPage();
    await restored.goto(url + hash);
    await restored.locator("#field-text mark").first().waitFor();
    assert.equal(await restored.locator(".finding-card.active").count(), 1);

    const mobile = await browser.newPage({ viewport: { width: 390, height: 844 } });
    await mobile.goto(url);
    assert.ok(await mobile.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    await fs.rm(dir, { recursive: true, force: true });
  }
});
