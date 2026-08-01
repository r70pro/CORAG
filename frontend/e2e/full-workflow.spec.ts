import { expect, test, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { basename, extname } from "node:path";

const runFullWorkflow = process.env.KIRAG_RUN_FULL_E2E === "1";
const sourcePdf =
  process.env.KIRAG_E2E_PDF || "/home/owner/Downloads/Docling_test_file.pdf";
const sourcePdfName = basename(sourcePdf);
const sourceStem = basename(sourcePdf, extname(sourcePdf));
const sourceMarkdownName = `0_${sourceStem}.md`;
const expectedPageCount = Number(process.env.KIRAG_E2E_EXPECTED_PAGES || "9");
const reuseLatestOcrRun = process.env.KIRAG_E2E_REUSE_OCR === "1";
const reuseReadyOcrRole = process.env.KIRAG_E2E_REUSE_OCR_ROLE === "1";
const ocrModel = "allenai/olmOCR-2-7B-1025-FP8";
const qwenModel =
  process.env.KIRAG_E2E_QWEN_MODEL || "Qwen/Qwen3.6-35B-A3B";
const reuseReadyQwen = process.env.KIRAG_E2E_REUSE_QWEN === "1";

function captureBrowserErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("requestfailed", (request) => {
    const reason = request.failure()?.errorText || "unknown";
    if (!reason.includes("ERR_ABORTED")) {
      errors.push(`requestfailed: ${request.url()} (${reason})`);
    }
  });
  return errors;
}

async function openWorkspace(page: Page, name: string) {
  await page.getByRole("button", { name, exact: true }).click();
}

test.describe.serial("real medicolegal workflow", () => {
  test.skip(!runFullWorkflow, "Set KIRAG_RUN_FULL_E2E=1 to mutate real local services and models");
  test.setTimeout(2 * 60 * 60 * 1000);

  test("OCR, inspect, embed, and manage the resulting case", async ({ page }) => {
    const browserErrors = captureBrowserErrors(page);
    await page.goto("/", { waitUntil: "networkidle" });

    if (!reuseLatestOcrRun) {
      if (!reuseReadyOcrRole) {
        const inferenceToggle = page.getByRole("button", { name: /Dedicated vLLM Roles/ });
        if (!(await page.getByText("OCR provisioning settings (analysis is managed independently by the production stack).").isVisible())) {
          await inferenceToggle.click();
        }
        await page.getByLabel("Model Name").first().selectOption(ocrModel);
        await page.getByRole("button", { name: "Recreate OCR", exact: true }).click();
        await expect(page.getByText(/Container created and started successfully/)).toBeVisible({
          timeout: 2 * 60 * 1000,
        });
      }
      await expect.poll(async () => page.evaluate(async () => {
        const response = await fetch("/api/docker/status");
        return response.ok ? ((await response.json()) as { status: string }).status : `http-${response.status}`;
      }), { timeout: 30 * 60 * 1000, intervals: [2_000, 5_000, 10_000] }).toBe("ready");

      await expect(page.getByRole("heading", { name: /Ingestion Pipeline/ })).toBeVisible();
      await page.locator("#ocr-model").selectOption(ocrModel);
      await page.locator("#pdf-file-input").setInputFiles(sourcePdf);
      await expect(page.getByText("1 file(s) selected", { exact: true })).toBeVisible();
      await expect(page.getByText(new RegExp(`${sourcePdfName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")} \\(\\d+(?:\\.\\d+)? MB\\)`))).toBeVisible();

      await page.getByRole("button", { name: "Start Batch Processing", exact: true }).click();
      await expect(page.getByText("● Processing", { exact: true })).toBeVisible();
      await expect(page.getByText("[Complete] Pipeline batch processing finished.", { exact: true })).toBeVisible({
        timeout: 45 * 60 * 1000,
      });
      await expect(page.getByText("● Completed", { exact: true })).toBeVisible();
      await expect(page.getByText("Completed Pages", { exact: true }).locator("..").getByText(String(expectedPageCount), { exact: true })).toBeVisible();
      await expect(page.getByText("Failed Pages", { exact: true }).locator("..").getByText("0", { exact: true })).toBeVisible();
      await expect(page.getByText("Done", { exact: true })).toBeVisible();
    }

    await openWorkspace(page, "🔍 Layout Inspector");
    await expect(page.getByRole("heading", { name: /Layout Inspector/ })).toBeVisible();
    const runSelector = page.getByText("📄 Select Processed Document Run", { exact: true }).locator("..").locator("select");
    await expect(runSelector).not.toHaveValue("");
    const selectedOcrRun = await page.evaluate(async (expectedMarkdownName) => {
      const response = await fetch("/api/documents/runs");
      if (!response.ok) throw new Error(`document runs HTTP ${response.status}`);
      const runs = (await response.json()) as { run_name: string; files: string[] }[];
      const selected = runs.find((run) => run.files.includes(expectedMarkdownName));
      if (!selected) throw new Error(`No completed ${expectedMarkdownName} OCR run was found`);
      return selected.run_name;
    }, sourceMarkdownName);
    await runSelector.selectOption(selectedOcrRun);
    await expect(page.locator('iframe[title="Source PDF Viewer"]')).toHaveAttribute("src", /\/api\/documents\/runs\/run_.*\/pdf/);
    const pdfValidation = await page.evaluate(async () => {
      const source = document.querySelector<HTMLIFrameElement>('iframe[title="Source PDF Viewer"]')?.src;
      if (!source) throw new Error("PDF iframe has no source");
      const response = await fetch(source, { headers: { Range: "bytes=0-15" } });
      return {
        status: response.status,
        contentType: response.headers.get("content-type"),
        signature: new TextDecoder().decode(await response.arrayBuffer()),
      };
    });
    expect([200, 206]).toContain(pdfValidation.status);
    expect(pdfValidation.contentType).toContain("application/pdf");
    expect(pdfValidation.signature).toContain("%PDF-");
    await expect(page.getByRole("heading", { name: /Raw Markdown Output/ })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Rendered Preview/ })).toBeVisible();
    await page.getByRole("button", { name: "Full Document", exact: true }).click();
    await expect(page.getByRole("button", { name: "Full Document", exact: true })).toHaveClass(/bg-indigo-600/);
    await page.getByRole("button", { name: "Page-by-Page", exact: true }).click();
    await page.getByRole("button", { name: "Next Page ➡️", exact: true }).click();
    await expect(page.getByText(`2 / ${expectedPageCount}`, { exact: true }).first()).toBeVisible();

    const runName = await runSelector.inputValue();
    const runFiles = await page.evaluate(async (name) => {
      const response = await fetch(`/api/documents/runs/${encodeURIComponent(name)}/files`);
      if (!response.ok) throw new Error(`run files HTTP ${response.status}`);
      return (await response.json()) as string[];
    }, runName);
    expect(runFiles).toContain(sourceMarkdownName);
    const markdown = await page.evaluate(async ({ name, filename }) => {
      const response = await fetch(
        `/api/documents/runs/${encodeURIComponent(name)}/markdown/${encodeURIComponent(filename)}`,
      );
      if (!response.ok) throw new Error(`markdown HTTP ${response.status}`);
      return response.text();
    }, { name: runName, filename: runFiles[0] });
    expect(markdown.length).toBeGreaterThan(500);

    await openWorkspace(page, "🧠 Embedding Pipeline");
    await expect(page.getByRole("heading", { name: /Embedding Pipeline/ })).toBeVisible();
    await page.getByLabel("Compute Engine Device").selectOption("auto");
    await page.getByLabel("Embedding Model Name").selectOption("BAAI/bge-large-en-v1.5");
    await page.getByRole("button", { name: "Save Configuration", exact: true }).click();
    await expect(page.getByText(/Embedding configuration saved successfully/)).toBeVisible();
    await page.getByLabel("Select OCR Run to Index").selectOption(runName);
    await page.getByRole("button", { name: "Index Selected Run", exact: true }).click();
    await expect(page.getByText(/Successfully indexed/).first()).toBeVisible({
      timeout: 30 * 60 * 1000,
    });
    await page.getByRole("button", { name: "Refresh", exact: true }).click();
    await expect(page.getByText(/\d+ Points/)).not.toHaveText("0 Points");
    const ocrRunId = await page.evaluate(async (selectedRunName) => {
      const response = await fetch("/api/pipeline/runs");
      if (!response.ok) throw new Error(`pipeline runs HTTP ${response.status}`);
      const runs = (await response.json()) as { run_dir: string; run_id: string }[];
      const selected = runs.find((run) => run.run_dir === selectedRunName);
      if (!selected) throw new Error(`Indexed run ${selectedRunName} was not returned by pipeline runs`);
      return selected.run_id;
    }, runName);

    const casesBefore = await page.evaluate(async () => {
      const response = await fetch("/api/case-summary");
      if (!response.ok) throw new Error(`case summary HTTP ${response.status}`);
      const data = (await response.json()) as { indexed_cases?: { run_id: string }[] };
      return (data.indexed_cases || []).map((item) => item.run_id);
    });
    await page.getByLabel("Select Markdown Files (.md)").setInputFiles({
      name: "e2e-deletion-case.md",
      mimeType: "text/markdown",
      buffer: Buffer.from(
        "# E2E Deletion Case\n\nPatient: Production Validation\n\nOn 2026-07-27, Dr Test recorded a temporary validation note.",
      ),
    });
    await page.getByLabel("New Case Name").fill(`E2E Deletion Case ${Date.now()}`);
    await page.getByRole("button", { name: "Upload & Index Markdown", exact: true }).click();
    await expect(page.getByText(/Successfully indexed/).last()).toBeVisible({
      timeout: 30 * 60 * 1000,
    });
    let temporaryRunId = "";
    await expect.poll(async () => {
      temporaryRunId = await page.evaluate(async (existing) => {
        const response = await fetch("/api/case-summary");
        if (!response.ok) return "";
        const data = (await response.json()) as { indexed_cases?: { run_id: string }[] };
        return (data.indexed_cases || []).map((item) => item.run_id).find((id) => !existing.includes(id)) || "";
      }, casesBefore);
      return temporaryRunId;
    }, { timeout: 60_000 }).not.toBe("");

    await openWorkspace(page, "📊 Case Dashboard");
    await page.getByRole("button", { name: "Refresh Dashboard", exact: true }).click();
    await expect(page.getByText(`📁 ${ocrRunId}`, { exact: true })).toBeVisible();
    await expect(page.getByText(/Active Case/).first()).toBeVisible();
    const temporaryRunIds = await page.evaluate(async () => {
      const response = await fetch("/api/pipeline/runs");
      if (!response.ok) throw new Error(`pipeline runs HTTP ${response.status}`);
      const runs = (await response.json()) as { run_dir: string; run_id: string }[];
      return runs.filter((run) => run.run_dir.startsWith("run_E2E_Deletion_Case_")).map((run) => run.run_id);
    });
    expect(temporaryRunIds).toContain(temporaryRunId);
    for (const runId of temporaryRunIds) {
      const temporaryCaseHeader = page.getByText(`📁 ${runId}`, { exact: true }).locator("..").locator("..");
      await temporaryCaseHeader.getByRole("checkbox").check();
    }
    const deleteTemporaryCases = page.getByRole("button", {
      name: `Delete Selected (${temporaryRunIds.length})`,
      exact: true,
    });
    await expect(deleteTemporaryCases).toBeEnabled();
    await deleteTemporaryCases.click();
    await expect.poll(async () => page.evaluate(async (deletedRunIds) => {
      const response = await fetch("/api/case-summary");
      if (!response.ok) return true;
      const data = (await response.json()) as { indexed_cases?: { run_id: string }[] };
      return (data.indexed_cases || []).some((item) => deletedRunIds.includes(item.run_id));
    }, temporaryRunIds)).toBe(false);
    await expect(page.getByRole("button", { name: /Delete Selected \(0\)/ })).toBeDisabled();

    expect(browserErrors).toEqual([]);
  });

  test("switch to Qwen, generate and export a timeline, then validate diagnostics", async ({ page }) => {
    const browserErrors = captureBrowserErrors(page);
    await page.goto("/", { waitUntil: "networkidle" });

    if (!reuseReadyQwen) {
      const inferenceToggle = page.getByRole("button", { name: /Inference Server/ });
      if (!(await page.getByText("Manage the local GPU inference container.").isVisible())) {
        await inferenceToggle.click();
      }
      await page.getByLabel("Model Name").first().selectOption(qwenModel);
      await page.getByRole("button", { name: "Recreate & Run", exact: true }).click();
      await expect(page.getByText(/Container created and started successfully/)).toBeVisible({ timeout: 2 * 60 * 1000 });
    }

    await expect.poll(async () => {
      return page.evaluate(async () => {
        const response = await fetch("/api/docker/status");
        return response.ok ? ((await response.json()) as { status: string }).status : `http-${response.status}`;
      });
    }, { timeout: 90 * 60 * 1000, intervals: [2_000, 5_000, 10_000] }).toBe("ready");

    // The full-precision Qwen model occupies most unified GPU memory; keep the
    // embedding and reranking stages deterministic on CPU for this workload.
    await openWorkspace(page, "🧠 Embedding Pipeline");
    await page.getByLabel("Compute Engine Device").selectOption("cpu");
    await page.getByRole("button", { name: "Save Configuration", exact: true }).click();
    await expect(page.getByText(/Embedding configuration saved successfully/)).toBeVisible();

    await openWorkspace(page, "💬 RAG Processing");
    await expect(page.getByRole("heading", { name: /RAG Processing/ })).toBeVisible();
    const modelNameInput = page.getByLabel("Model Name").last();
    await expect(modelNameInput).toHaveValue(qwenModel);
    if (await modelNameInput.isEnabled()) await modelNameInput.fill(qwenModel);
    const rerankerDevice = page.getByLabel("Reranker Device");
    await rerankerDevice.selectOption("cpu");
    await expect(rerankerDevice).toHaveValue("cpu");
    await page.getByRole("button", { name: "Save Model Settings", exact: true }).click();
    await expect(page.getByText(/Settings saved successfully/)).toBeVisible();
    await expect.poll(async () => page.evaluate(async () => {
      const response = await fetch("/api/settings");
      return response.ok
        ? ((await response.json()) as { reranker_device?: string }).reranker_device
        : `http-${response.status}`;
    })).toBe("cpu");
    const rerankerToggle = page.getByLabel("Enable Cross-Encoder Reranker");
    await expect(rerankerToggle).toBeChecked();
    await rerankerToggle.uncheck();
    await page.getByRole("button", { name: "📋 Timeline", exact: true }).click();
    await page.getByLabel("Maximum Output Tokens").fill("2048");
    const activeCaseSelector = page.locator("select").filter({ has: page.locator("option", { hasText: "Select Active Case Context" }) }).first();
    const caseOptions = await activeCaseSelector.locator("option").count();
    expect(caseOptions).toBeGreaterThan(1);
    const indexedOcrCase = await page.evaluate(async (expectedMarkdownName) => {
      const [documentsResponse, pipelineResponse] = await Promise.all([
        fetch("/api/documents/runs"),
        fetch("/api/pipeline/runs"),
      ]);
      if (!documentsResponse.ok || !pipelineResponse.ok) {
        throw new Error("Unable to resolve the indexed OCR case");
      }
      const documentRuns = (await documentsResponse.json()) as { run_name: string; files: string[] }[];
      const pipelineRuns = (await pipelineResponse.json()) as { run_dir: string; run_id: string; is_indexed?: boolean }[];
      const selectedDocument = documentRuns.find((run) => run.files.includes(expectedMarkdownName));
      const selectedCase = pipelineRuns.find(
        (run) => run.run_dir === selectedDocument?.run_name && run.is_indexed !== false,
      );
      if (!selectedCase) throw new Error("The completed OCR run is not indexed");
      return selectedCase.run_id;
    }, sourceMarkdownName);
    await activeCaseSelector.selectOption(indexedOcrCase);
    await expect(activeCaseSelector).toHaveValue(indexedOcrCase);

    const prompt = page.getByPlaceholder("Ask a medicolegal question or request an audit...").last();
    await prompt.fill(
      "Output only a Markdown timeline table, beginning with a header row containing Date, Event, Provider/Author, and Source columns. Include the ten earliest material dated events in chronological order, exact original-PDF page provenance, and source-supported verification details only.",
    );
    const sendButton = page.getByRole("button", { name: "Send Query", exact: true }).last();
    await sendButton.click();
    await expect(sendButton).toBeDisabled();
    const answer = page.locator(".whitespace-pre-wrap").last();
    await expect(answer).toContainText(/Date/i, { timeout: 30 * 60 * 1000 });
    const stopButton = page.getByRole("button", { name: "Stop generating", exact: true }).last();
    if (await stopButton.isEnabled()) await stopButton.click();
    const answerText = await answer.innerText();
    expect(answerText).toMatch(/Date/i);
    expect(answerText).toMatch(/Event/i);
    expect(answerText).toMatch(/Provider|Author/i);
    expect(answerText).toMatch(/Source/i);
    expect(answerText).toContain(sourcePdfName);
    expect(answerText).toMatch(/\b(?:p\.|pp\.|page(?:s)?)\s*\d+/i);
    expect(answerText).not.toMatch(/No relevant document excerpts/i);
    expect(answerText).not.toMatch(/thinking process/i);
    expect(answerText).not.toMatch(/\[Source\s+\d+\]/i);
    expect(answerText).not.toMatch(/Error processing query/i);
    expect(answerText).not.toMatch(/Incomplete response/i);

    for (const buttonName of ["Export MD", "TXT", "CSV", "DOCX", "Timeline DOCX"]) {
      const downloadPromise = page.waitForEvent("download");
      await page.getByRole("button", { name: buttonName, exact: true }).last().click();
      const download = await downloadPromise;
      const filename = download.suggestedFilename();
      expect(filename).toMatch(/\.(md|txt|csv|docx)$/);
      const downloadPath = await download.path();
      expect(downloadPath).not.toBeNull();
      const contents = await readFile(downloadPath!);
      expect(contents.length).toBeGreaterThan(20);
      if (filename.endsWith(".docx")) {
        expect(contents.subarray(0, 2).toString()).toBe("PK");
      } else {
        const text = contents.toString("utf8");
        expect(text).toMatch(/Date|Timeline|medicolegal|No relevant/i);
        if (filename.endsWith(".csv")) expect(text).toContain(",");
      }
    }

    await openWorkspace(page, "🖥️ System Diagnostics");
    await page.getByTitle("Refresh Diagnostic Telemetry").click();
    for (const serviceName of ["POSTGRES", "REDIS", "MINIO", "QDRANT", "VLLM_OCR", "VLLM_ANALYSIS"]) {
      await expect(page.getByRole("heading", { name: serviceName, exact: true })).toBeVisible();
    }
    await expect(page.getByText("6/6 Online", { exact: true })).toBeVisible();
    await expect(page.getByText("NVIDIA GB10", { exact: true })).toBeVisible();
    await page.getByTitle("Refresh Installed Models List").click();
    await expect(page.getByRole("row", { name: new RegExp(`${qwenModel}.*ACTIVE`) })).toBeVisible();
    await page.getByTitle("Refresh Container Logs").click();
    await expect(page.getByRole("button", { name: /vLLM Logs \((READY|RUNNING)\)/ })).toBeVisible();

    expect(browserErrors).toEqual([]);
  });

  test("diagnostics inventory and live container telemetry", async ({ page }) => {
    const browserErrors = captureBrowserErrors(page);
    await page.goto("/", { waitUntil: "networkidle" });
    await openWorkspace(page, "🖥️ System Diagnostics");
    await page.getByTitle("Refresh Diagnostic Telemetry").click();
    for (const serviceName of ["POSTGRES", "REDIS", "MINIO", "QDRANT", "VLLM_OCR", "VLLM_ANALYSIS"]) {
      await expect(page.getByRole("heading", { name: serviceName, exact: true })).toBeVisible();
    }
    await expect(page.getByText("6/6 Online", { exact: true })).toBeVisible();
    await expect(page.getByText("NVIDIA GB10", { exact: true })).toBeVisible();
    await page.getByTitle("Refresh Installed Models List").click();
    await expect(page.getByRole("row", { name: new RegExp(`${qwenModel}.*ACTIVE`) })).toBeVisible();
    await page.getByTitle("Refresh Container Logs").click();
    await expect(page.getByRole("button", { name: /vLLM Logs \((READY|RUNNING)\)/ })).toBeVisible();
    expect(browserErrors).toEqual([]);
  });
});
