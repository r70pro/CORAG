import { defineConfig, devices } from "@playwright/test";

const browserExecutable = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
const baseURL = process.env.PLAYWRIGHT_BASE_URL || "http://127.0.0.1:3000";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: "line",
  timeout: 2 * 60 * 60 * 1000,
  use: {
    baseURL,
    actionTimeout: 30_000,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    launchOptions: browserExecutable ? { executablePath: browserExecutable } : {},
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "npm run start -- --hostname 127.0.0.1",
    url: baseURL,
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
