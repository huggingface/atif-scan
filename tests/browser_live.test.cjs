/*
 * Optional, synthetic-only live frontend test (no downloads or model calls):
 * NODE_PATH=/home/ssmith/source/harbor-hf/node_modules:/home/ssmith/source/inspector/node_modules \
 *   node --test tests/browser_live.test.cjs
 * BROWSER_EXECUTABLE may select an already-installed Chromium.
 */
"use strict";
const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const { spawn } = require("node:child_process");

const root = path.resolve(__dirname, "..");
const credential = "sk-proj-syntheticCredential8274";
const html = '<img src=x onerror="window.syntheticInjected=true">';
const message = `Synthetic credential: ${credential}\n` + "🌿漢é".repeat(2600) + html;
const masked = Array.from(message.replace(credential, "***"));
const check = "observation.credentials_exposed";

function installedPlaywright() {
  try {
    return require("playwright");
  } catch (error) {
    if (error.code !== "MODULE_NOT_FOUND" || !error.message.includes("'playwright'")) throw error;
    return null;
  }
}

// Own the entire uv/CLI process group, including startup failures.
function startServer(directory) {
  const child = spawn("uv", [
    "run", "atif-scan", path.join(directory, "trajectory.json"),
    "--browse", "--task", "synthetic-browser", "--packs", "none", "--no-cache",
    "--feedback-dir", path.join(directory, "feedback"),
  ], {
    cwd: root, detached: true, stdio: ["ignore", "ignore", "pipe"],
    env: { ...process.env, ATIF_SCAN_HOME: path.join(directory, "home") },
  });
  const exited = new Promise(resolve => child.once("close", resolve));
  const ready = new Promise((resolve, reject) => {
    let stderr = "";
    const timer = setTimeout(() => reject(new Error("Browser CLI startup timed out (stderr withheld).")), 30000);
    const finish = (error, url) => {
      clearTimeout(timer);
      if (error) reject(error);
      else resolve(url);
    };
    child.on("error", () => finish(new Error("Could not spawn uv (details withheld).")));
    child.once("exit", () => finish(new Error("Browser CLI exited before startup (stderr withheld).")));
    child.stderr.on("data", chunk => {
      stderr = (stderr + chunk.toString()).slice(-8192);
      const match = stderr.match(/http:\/\/127\.0\.0\.1:\d+\/#token=[A-Za-z0-9_-]+(?=\s)/);
      if (match) finish(null, match[0]);
    });
  });
  async function stop() {
    const signal = name => {
      if (!child.pid) return;
      try { process.kill(-child.pid, name); }
      catch (error) { if (error.code !== "ESRCH") throw error; }
    };
    signal("SIGINT");
    const timer = setTimeout(() => signal("SIGKILL"), 3000);
    try { await exited; }
    finally { clearTimeout(timer); signal("SIGKILL"); }
  }
  return { ready, stop };
}

async function textEquals(page, selector, text) {
  await page.waitForFunction(({ selector, text }) =>
    document.querySelector(selector)?.textContent === text, { selector, text });
}

async function selectCredential(page) {
  await page.locator("#priority").selectOption("low");
  await page.locator("#status").selectOption("match");
  await page.locator("#trial-list button").first().click();
  await page.locator("#finding-list button").filter({ hasText: check }).click();
  await textEquals(page, "#feedback-state", "Saved feedback loaded.");
  await page.locator("#locations button").first().click();
  await textEquals(page, "#field-text", masked.slice(0, 3000).join(""));
  await textEquals(page, "#review-heading", "Possible credential-like value");
  assert.match(await page.locator("#finding-meta").textContent(), /Priority: low · status: match/);
  assert.equal(await page.locator("#offset").inputValue(), "0");
  assert.equal(await page.locator("#field-text mark").count(), 1);
  assert.equal(await page.locator("#field-text mark").textContent(), "credential: ***");
  assert.ok(!(await page.locator("#field-text").textContent()).includes(credential));
  await textEquals(page, "#focus-note",
    "Expanded masked region, not the exact trigger span. Highlight shows the portion on this page.");
}

async function unknownFindings(page) {
  await page.locator("#priority").selectOption("all");
  await page.locator("#status").selectOption("unknown");
  const rows = await page.locator("#finding-list button").allTextContents();
  assert.ok(rows.length > 0, "Synthetic missing evidence must produce unknown findings");
  const metadata = await page.locator("#finding-list button > span:last-child").allTextContents();
  assert.equal(metadata.length, rows.length);
  assert.ok(metadata.every(row => / · unknown · /.test(row)));
  assert.ok(rows.every(row => row.includes("Check: ")));
  return rows;
}

async function defaultFilters(page) {
  assert.equal(await page.locator("#priority").inputValue(), "medium+");
  assert.equal(await page.locator("#status").inputValue(), "match");
  await textEquals(page, "#trial-count", "0 / 1");
  assert.equal(await page.locator("#trial-list button").count(), 0);
  assert.equal(await page.locator("#finding-list button").filter({ hasText: check }).count(), 0,
    "Low credential finding must not appear in the default medium+ queue");
  assert.equal(await page.locator("#finding-list button").count(), 0);
}

test("synthetic live desk: masked evidence, Unicode, literal HTML, durable explicit feedback and gaps", {
  timeout: 120000,
}, async t => {
  const playwright = installedPlaywright();
  if (!playwright) {
    t.skip("Optional: install nothing here; point NODE_PATH at an existing Playwright installation.");
    return;
  }
  const executable = process.env.BROWSER_EXECUTABLE;
  const available = executable || playwright.chromium.executablePath();
  try { await fs.access(available); }
  catch (error) {
    if (error.code !== "ENOENT" || executable) throw error;
    t.skip("Optional: Chromium is absent; set BROWSER_EXECUTABLE to an existing installation. No download attempted.");
    return;
  }
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "atif-browser-synthetic-"));
  let server, browser, launchURL = "", phase = "startup";
  try {
    await fs.writeFile(path.join(directory, "trajectory.json"), JSON.stringify({
      steps: [
        { step_id: 7, source: "agent", message },
        { step_id: 20, source: "agent", tool_calls: [
          { tool_call_id: "synthetic-missing", function_name: "synthetic_unavailable", arguments: {} },
        ] },
      ],
    }), { mode: 0o600 });
    server = startServer(directory);
    launchURL = await server.ready;
    browser = await playwright.chromium.launch({ headless: true, ...(executable ? { executablePath: executable } : {}) });
    const context = await browser.newContext({ viewport: { width: 1500, height: 1050 } });
    const page = await context.newPage();
    page.setDefaultTimeout(10000);
    let origin = new URL(launchURL).origin;
    let externalRequests = 0, pageErrors = 0, feedbackWrites = 0;
    await context.route("**/*", route => {
      if (new URL(route.request().url()).origin !== origin) {
        externalRequests++;
        return route.abort();
      }
      return route.continue();
    });
    page.on("pageerror", () => pageErrors++);
    page.on("request", request => {
      if (new URL(request.url()).pathname === "/api/feedback") feedbackWrites++;
    });
    // Auth rejection without sending or printing the token.
    const unauthorized = await context.request.post(`${origin}/api/overview`, { data: {} });
    assert.equal(unauthorized.status(), 403);
    await page.goto(launchURL);
    phase = "default filters and masked finding";
    await defaultFilters(page);
    await selectCredential(page);
    const coverage = await page.locator("#coverage").textContent();
    assert.match(coverage, /coverage_gaps/);
    assert.match(await page.locator("#trial-title").textContent(), /reward unknown/);
    const unknown = await unknownFindings(page);
    await page.locator("#priority").selectOption("low");
    await page.locator("#status").selectOption("match");

    phase = "Unicode paging and literal search";
    await page.locator("#next-page").click();
    await textEquals(page, "#field-text", masked.slice(3000, 6000).join(""));
    assert.equal(await page.locator("#offset").inputValue(), "3000");
    await page.locator("#previous-page").click();
    await textEquals(page, "#field-text", masked.slice(0, 3000).join(""));
    await page.locator("#search-query").fill(html);
    await page.locator("#search-button").click();
    await page.locator("#search-results button").waitFor();
    assert.equal(await page.locator("#search-results button").count(), 1);
    await page.locator("#search-results button").click();
    await textEquals(page, "#field-text", html);
    assert.equal(Number(await page.locator("#offset").inputValue()), masked.length - Array.from(html).length);
    assert.equal(await page.locator("#field-text img").count(), 0);
    assert.equal(await page.evaluate(() => window.syntheticInjected), undefined);
    assert.match(await page.locator("#search-state").textContent(), /not proof of absence/);

    phase = "explicit save and reload";
    await page.locator("#verdict").selectOption("false_positive");
    await page.locator("#note").fill("Synthetic unsaved draft 🌿");
    await textEquals(page, "#feedback-state", "Unsaved feedback — use Save feedback to persist.");
    assert.equal(feedbackWrites, 0);
    await page.reload();
    await selectCredential(page);
    assert.equal(await page.locator("#note").inputValue(), "");
    assert.equal(await page.locator("#verdict").inputValue(), "unreviewed");
    const note = "Synthetic review only 🌿漢é";
    await page.locator("#verdict").selectOption("false_positive");
    await page.locator("#note").fill(note);
    await page.locator("#save").click();
    await textEquals(page, "#feedback-state", "Feedback saved for this finding.");
    assert.equal(feedbackWrites, 1);
    const meta = await page.locator("#finding-meta").textContent();
    assert.match(meta, /Priority: low · status: match/);
    await page.reload();
    await selectCredential(page);
    assert.equal(await page.locator("#note").inputValue(), note);
    assert.equal(await page.locator("#verdict").inputValue(), "false_positive");

    phase = "process restart with identical source and feedback";
    await server.stop();
    server = startServer(directory);
    launchURL = await server.ready;
    origin = new URL(launchURL).origin;
    await page.goto(launchURL);
    await defaultFilters(page);
    await selectCredential(page);
    assert.equal(await page.locator("#note").inputValue(), note);
    assert.equal(await page.locator("#verdict").inputValue(), "false_positive");
    assert.equal(await page.locator("#finding-meta").textContent(), meta);
    assert.doesNotMatch(await page.locator("#finding-meta").textContent(), /Priority: medium/,
      "Full restart must not restore stale medium credential priority");
    assert.equal(await page.locator("#coverage").textContent(), coverage);
    assert.deepEqual(await unknownFindings(page), unknown);

    phase = "mobile layout";
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(directory, "synthetic-mobile.png"), fullPage: true });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
      "Mobile document must not overflow horizontally");
    assert.equal(externalRequests, 0, "No external requests, including HTML fixture URLs");
    assert.equal(pageErrors, 0, "No frontend exceptions");
  } catch (error) {
    // Playwright call logs can include navigation URLs; never emit the bearer token.
    const detail = String(error.message).replace(/http:\/\/127\.0\.0\.1:\d+\/[^\s"'<>)]*/g, "[private browser URL]");
    throw new Error(`${phase}: ${detail}`);
  } finally {
    try { await browser?.close(); }
    finally {
      try { await server?.stop(); }
      finally { await fs.rm(directory, { recursive: true, force: true }); }
    }
  }
});
