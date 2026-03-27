import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(path.join(here, "..", ".."));
const resultsDir = path.join(root, "results");
const outDir = path.join(root, "dashboard", "public", "data");
const outFile = path.join(outDir, "dashboard_bundle.json");

function readJson(filePath, fallback = null) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return fallback;
  }
}

function mean(nums) {
  if (!nums.length) return null;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function aggregateMode(records, modeName) {
  if (!Array.isArray(records) || records.length === 0) {
    return { mode: modeName, mem_gb: null, ttft_ms: null, tok_per_sec: null, quality: null };
  }
  const mem = records.map((r) => r.mem_gb).filter((v) => typeof v === "number");
  const ttft = records.map((r) => r.ttft_ms).filter((v) => typeof v === "number");
  const tps = records.map((r) => r.tok_per_sec).filter((v) => typeof v === "number");
  const quality = records.map((r) => r.quality_score).filter((v) => typeof v === "number");
  return {
    mode: modeName,
    mem_gb: mean(mem),
    ttft_ms: mean(ttft),
    tok_per_sec: mean(tps),
    quality: quality.length ? mean(quality) : null
  };
}

function latestDir(base, prefix) {
  if (!fs.existsSync(base)) return null;
  const dirs = fs
    .readdirSync(base, { withFileTypes: true })
    .filter((d) => d.isDirectory() && d.name.startsWith(prefix))
    .map((d) => path.join(base, d.name));
  if (!dirs.length) return null;
  dirs.sort((a, b) => fs.statSync(a).mtimeMs - fs.statSync(b).mtimeMs);
  return dirs[dirs.length - 1];
}

function parseJsonl(filePath, maxItems = 120) {
  if (!fs.existsSync(filePath)) return [];
  const lines = fs
    .readFileSync(filePath, "utf-8")
    .split(/\r?\n/)
    .filter(Boolean);
  return lines
    .slice(-maxItems)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter(Boolean)
    .reverse();
}

function buildBundle() {
  const fp16 = readJson(path.join(resultsDir, "fp16_results.json"), []);
  const bit8 = readJson(path.join(resultsDir, "8bit_results.json"), []);
  const bit4 = readJson(path.join(resultsDir, "4bit_results.json"), []);

  const comparisonRows = [
    aggregateMode(fp16, "FP16"),
    aggregateMode(bit8, "8-bit"),
    aggregateMode(bit4, "4-bit")
  ];

  const scatterRows = comparisonRows
    .filter((r) => typeof r.mem_gb === "number")
    .map((r) => ({
      mode: r.mode,
      mem_gb: Number(r.mem_gb.toFixed(3)),
      quality: r.quality ?? 0,
      tok_per_sec: r.tok_per_sec ?? 0
    }));

  const loadRoot = path.join(resultsDir, "load-testing");
  const latestLoadRun = latestDir(loadRoot, "run_");
  const latencyRows = latestLoadRun
    ? (readJson(path.join(latestLoadRun, "load_test_summary.json"), { results: [] }).results || [])
        .map((r) => ({
          users: r.users,
          generate_avg_ms: r.generate_avg_ms,
          generate_p95_ms: r.generate_p95_ms,
          generate_rps: r.generate_rps
        }))
        .filter((r) => typeof r.users === "number")
    : [];

  const routingRows = parseJsonl(path.join(resultsDir, "routing_log.jsonl"));

  return {
    generatedAt: new Date().toISOString(),
    comparisonRows,
    scatterRows,
    latencyRows,
    routingRows
  };
}

fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(outFile, JSON.stringify(buildBundle(), null, 2), "utf-8");
console.log(`[dashboard] data synced -> ${outFile}`);

