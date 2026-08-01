import { expect, test, type Page } from "@playwright/test";

function failOnBrowserErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("requestfailed", (request) => {
    const reason = request.failure()?.errorText || "unknown";
    // Switching workspaces unloads the PDF iframe and Chromium reports the
    // intentionally cancelled range request as ERR_ABORTED.
    if (!reason.includes("ERR_ABORTED")) {
      errors.push(`requestfailed: ${request.url()} (${reason})`);
    }
  });
  return errors;
}

test("hydrates and switches every workspace from both navigation bars", async ({ page }) => {
  const browserErrors = failOnBrowserErrors(page);
  await page.goto("/", { waitUntil: "networkidle" });

  const sidebarViews = [
    ["📥 Ingestion Pipeline", "Ingestion Pipeline"],
    ["🔍 Layout Inspector", "Layout Inspector"],
    ["🧠 Embedding Pipeline", "Embedding Pipeline"],
    ["📊 Case Dashboard", "Case Dashboard"],
    ["💬 RAG Processing", "RAG Processing"],
    ["🖥️ System Diagnostics", "System Diagnostics"],
  ] as const;

  for (const [buttonName, heading] of sidebarViews) {
    await page.getByRole("button", { name: buttonName, exact: true }).click();
    await expect(page.getByRole("heading", { name: heading, exact: false })).toBeVisible();
  }

  await page.getByRole("button", { name: "Ingestion", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Ingestion Pipeline", exact: false })).toBeVisible();
  await page.getByRole("button", { name: "RAG Chat", exact: true }).click();
  await expect(page.getByRole("heading", { name: "RAG Processing", exact: false })).toBeVisible();

  expect(browserErrors).toEqual([]);
});

test("safe controls inside every workspace respond to clicks", async ({ page }) => {
  const browserErrors = failOnBrowserErrors(page);
  await page.goto("/", { waitUntil: "networkidle" });

  await page.getByRole("button", { name: "Advanced Parameters", exact: true }).click();
  await expect(page.getByText("Max Concurrent Requests", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Advanced Parameters", exact: true }).click();
  await expect(page.getByText("Max Concurrent Requests", { exact: true })).toBeHidden();

  await page.getByRole("button", { name: "🔍 Layout Inspector", exact: true }).click();
  await page.getByRole("button", { name: "Full Document", exact: true }).first().click();
  await expect(page.getByRole("button", { name: "Full Document", exact: true }).first()).toHaveClass(/bg-(cyan|indigo)-600/);
  await page.getByRole("button", { name: "Page View", exact: true }).click();
  await expect(page.getByRole("button", { name: "Page View", exact: true })).toHaveClass(/bg-cyan-600/);

  await page.getByRole("button", { name: "🧠 Embedding Pipeline", exact: true }).click();
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect(page.getByText("Qdrant ● Connected", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "📊 Case Dashboard", exact: true }).click();
  await page.getByRole("button", { name: "Refresh Dashboard", exact: true }).click();
  await page.getByRole("button", { name: "Select All", exact: true }).click();
  await page.getByRole("button", { name: "Clear Selection", exact: true }).click();

  await page.getByRole("button", { name: "💬 RAG Processing", exact: true }).click();
  await page.getByRole("button", { name: "Hide Panel", exact: true }).click();
  await expect(page.getByRole("button", { name: "Show Panel", exact: true })).toBeVisible();
  for (const mode of [
    "💬 Free Q&A",
    "📋 Timeline",
    "🏥 Injury Summary",
    "🔍 Inconsistency Finder",
    "💊 Medication Tracker",
  ]) {
    const modeButton = page.getByRole("button", { name: mode, exact: true });
    await modeButton.click();
    await expect(modeButton).toHaveClass(/bg-indigo-600/);
  }
  await page.getByRole("button", { name: "Show Panel", exact: true }).click();
  await expect(page.getByText("📦 Document Indexing", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "🖥️ System Diagnostics", exact: true }).click();
  await page.getByRole("button", { name: /Diagnostic System Log/, exact: false }).click();
  await expect(page.getByRole("button", { name: /Diagnostic System Log/, exact: false })).toHaveClass(/bg-indigo-600/);
  const vllmLogs = page.getByRole("button", { name: /vLLM Logs/, exact: false });
  await vllmLogs.click();
  await expect(vllmLogs).toHaveClass(/bg-indigo-600/);
  await page.getByTitle("Refresh Container Logs").click();

  expect(browserErrors).toEqual([]);
});

test("interactive shell controls update visible state without browser errors", async ({ page }) => {
  const browserErrors = failOnBrowserErrors(page);
  await page.goto("/", { waitUntil: "networkidle" });

  const inferenceToggle = page.getByRole("button", { name: /Dedicated vLLM Roles/ });
  await inferenceToggle.click();
  await expect(page.getByText("OCR provisioning settings (analysis is managed independently by the production stack).")).toBeHidden();
  await inferenceToggle.click();
  await expect(page.getByText("OCR provisioning settings (analysis is managed independently by the production stack).")).toBeVisible();

  await page.getByRole("button", { name: "Compact", exact: true }).click();
  const applicationShell = page.locator("body > div.flex").first();
  await expect(applicationShell).toHaveClass(/text-\[13px\]/);
  await page.getByRole("button", { name: "Comfortable", exact: true }).click();
  await expect(applicationShell).not.toHaveClass(/text-\[13px\]/);

  await page.getByTitle("GPU VRAM Memory Distribution").click();
  await expect(page.getByText("GPU VRAM Telemetry")).toBeVisible();
  await page.getByTitle("System & Backing Services Health").click();
  await expect(page.getByText("Services Health Matrix")).toBeVisible();

  const role = page.locator("aside select").last();
  await role.selectOption("Clinical Reviewer");
  await expect(role).toHaveValue("Clinical Reviewer");

  expect(browserErrors).toEqual([]);
});
