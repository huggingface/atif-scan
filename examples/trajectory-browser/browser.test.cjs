/* Optional real-browser smoke test; requires Playwright but no external services. */
const { chromium } = require("playwright");
const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const assets = new Map([
  ["/", ["index.html", "text/html"]], ["/browser.css", ["browser.css", "text/css"]],
  ["/fixture.js", ["fixture.js", "text/javascript"]], ["/evidence.js", ["evidence.js", "text/javascript"]],
  ["/browser.js", ["browser.js", "text/javascript"]],
]);
const server = http.createServer((req, res) => {
  const asset = assets.get(new URL(req.url, "http://127.0.0.1").pathname);
  if (!asset) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { "Content-Type": asset[1], "Cache-Control": "no-store" });
  res.end(fs.readFileSync(path.join(__dirname, asset[0])));
});
const assert = require("node:assert/strict");
let active;
(async () => {
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}/`;
  const browser = await chromium.launch({
    ...(process.env.BROWSER_EXECUTABLE ? { executablePath: process.env.BROWSER_EXECUTABLE } : {}),
    headless: true,
  });
  active = browser;
  const page = await browser.newPage({ viewport: { width: 1500, height: 1050 } });
  const errors = [], requests = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", req => requests.push(req.url()));
  await page.goto(base);
  await page.locator("#field-text mark").waitFor();
  assert.match(await page.locator("#field-text mark").innerText(), /DEMO SOURCE REFERENCE/);
  assert.match(await page.locator("#field-title").innerText(), /Step 70/);
  const initial = await page.locator("#page-status").innerText();

  await page.locator("#next-page").click();
  assert.notEqual(await page.locator("#page-status").innerText(), initial);
  await page.reload();
  assert.notEqual(await page.locator("#page-status").innerText(), initial);
  await page.getByRole("button", { name: "Task baseline · 10", exact: true }).click();
  assert.match(await page.locator("#field-text").innerText(), /Public research is permitted/);
  await page.getByRole("button", { name: "Returned reference · 70", exact: true }).click();
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await page.locator("#evidence-query").fill("DEMO SOURCE REFERENCE");
  await page.getByRole("button", { name: "Find", exact: true }).click();
  assert.equal(await page.locator(".search-hit").count(), 2);
  await page.locator(".search-hit").last().click();
  assert.match(await page.locator("#field-title").innerText(), /Step 90/);
  await page.locator(".trial-row").filter({ hasText: "index-maintenance" }).click();
  assert.equal(await page.locator(".search-hit").count(), 0, "Old-trial search matches must not leak into the new trial.");
  await page.getByRole("button", { name: "Findings", exact: true }).click();
  await page.getByRole("button", { name: "Absent result · 25", exact: true }).click();
  assert.match(await page.locator("#availability").innerText(), /receipt remains unknown/);
  assert.equal(await page.locator("#next-page").isDisabled(), true);
  await page.locator(".finding-select").filter({ hasText: "Result association inferred" }).click();
  await page.getByRole("button", { name: "Inferred result · 35", exact: true }).click();
  assert.match(await page.locator("#provenance.warning").innerText(), /unique_remainder/);
  await page.getByRole("button", { name: "Step 55: Unavailable image", exact: true }).click();
  assert.match(await page.locator("#availability").innerText(), /Media content unavailable/);
  await page.getByRole("button", { name: "Judge review", exact: true }).click();
  assert.match(await page.locator("#review-panel").innerText(), /Stale answer/);
  await page.getByRole("button", { name: "All trials", exact: true }).click();
  await page.locator(".trial-row").filter({ hasText: "local-summary" }).click();
  await page.getByRole("button", { name: "Step 8: Read local fixture", exact: true }).click();
  await page.locator(".step-card.active .field-button").filter({ hasText: "result" }).click();
  assert.match(await page.locator("#field-text").innerText(), /<img src=x onerror=/);
  assert.equal(await page.locator("#field-text img").count(), 0);
  assert.equal(await page.evaluate(() => window.demoInjected), undefined);
  await page.locator(".trial-row").filter({ hasText: "calibration-notes" }).click();
  await page.getByRole("button", { name: "Findings", exact: true }).click();
  await page.getByRole("button", { name: "Dark theme", exact: true }).click();

  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  assert.deepEqual(errors, []);
  assert.ok(requests.every(url => url.startsWith(base)), "No external network assets.");
  console.log("Browser smoke: deep focus, paging/reload, baseline/context, search, missing media/result, stale review, inferred pairing, literal HTML and responsive layout: PASS");
  await browser.close();
  server.close();
})().catch(async error => { console.error(error); await active?.close(); server.close(); process.exitCode = 1; });
