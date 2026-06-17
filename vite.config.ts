import { defineConfig } from "vite";
import { execSync } from "node:child_process";

// Build identifier shown in the app header so it's obvious which build is
// running. Uncommitted rebuilds keep the same git hash, so the timestamp is
// what actually distinguishes one build from the next.
function buildId(): string {
  let git = "nogit";
  try {
    const hash = execSync("git rev-parse --short HEAD").toString().trim();
    const dirty = execSync("git status --porcelain").toString().trim().length > 0;
    git = hash + (dirty ? "-dirty" : "");
  } catch {
    /* not a git checkout */
  }
  const now = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  const stamp = `${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${pad(now.getHours())}:${pad(now.getMinutes())}`;
  return `${git} · ${stamp}`;
}

export default defineConfig({
  define: {
    __BUILD_ID__: JSON.stringify(buildId()),
  },
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: {
      ignored: ["**/src-tauri/**"],
    },
  },
});
