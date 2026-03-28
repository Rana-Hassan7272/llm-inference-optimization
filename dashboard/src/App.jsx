import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis
} from "recharts";

const REFRESH_MS = 15000;

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function Card({ title, children, subtitle }) {
  return (
    <section className="card">
      <div className="card-header">
        <h2>{title}</h2>
        {subtitle ? <span>{subtitle}</span> : null}
      </div>
      {children}
    </section>
  );
}

function App() {
  const [bundle, setBundle] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        // Base static bundle (committed artifacts)
        const res = await fetch(`/data/dashboard_bundle.json?ts=${Date.now()}`, { cache: "no-store" });
        let data = {};
        if (res.ok) {
          data = await res.json();
          // Normalize latency rows if present
          if (Array.isArray(data.latencyRows)) {
            data.latencyRows = data.latencyRows.map((row) => ({
              users: Number(row.users),
              generate_avg_ms: Number(row.generate_avg_ms),
              generate_p95_ms: Number(row.generate_p95_ms),
              generate_rps: Number(row.generate_rps),
              generate_fail_ratio:
                row.generate_fail_ratio == null ? null : Number(row.generate_fail_ratio),
              total_rps: Number(row.total_rps),
            }));
          }
        } else {
          throw new Error(`Failed to load dashboard data (${res.status})`);
        }

        // Try to augment latency from a separate summary if bundle has none
        const hasLatency = Array.isArray(data.latencyRows) && data.latencyRows.length > 0;
        const tryLoadLatencyFrom = async (url) => {
          try {
            const ts = Date.now();
            const bust = url.includes("?") ? `&ts=${ts}` : `?ts=${ts}`;
            const l = await fetch(`${url}${bust}`, { mode: "cors", cache: "no-store" });
            if (!l.ok) return false;
            const summary = await l.json();
            const rowsSrc = Array.isArray(summary) ? summary
              : Array.isArray(summary?.results) ? summary.results
              : [];
            const latencyRows = rowsSrc.map((row) => ({
              users: Number(row.users),
              generate_avg_ms: Number(row.generate_avg_ms),
              generate_p95_ms: Number(row.generate_p95_ms),
              generate_rps: Number(row.generate_rps),
              generate_fail_ratio: row.generate_fail_ratio == null ? null : Number(row.generate_fail_ratio),
              total_rps: Number(row.total_rps),
            })).filter((r) => Number.isFinite(r.users));
            if (latencyRows.length) {
              data.latencyRows = latencyRows;
              return true;
            }
            return false;
          } catch {
            return false;
          }
        };

        if (!hasLatency) {
          const latencyUrl = import.meta.env.VITE_LATENCY_JSON_URL || "";
          if (latencyUrl) {
            await tryLoadLatencyFrom(latencyUrl);
          }
          if (!Array.isArray(data.latencyRows) || data.latencyRows.length === 0) {
            const candidates = [
              "/data/load_test_summary.json",
              "/data/latest/load_test_summary.json",
              "/data/latest.json",
            ];
            for (const url of candidates) {
              // eslint-disable-next-line no-await-in-loop
              const ok = await tryLoadLatencyFrom(url);
              if (ok) break;
            }
          }
          if (!Array.isArray(data.latencyRows) || data.latencyRows.length === 0) {
            setError("Latency dataset not found. Ensure /data/load_test_summary.json exists in the deployed site.");
          }
        }

        // Optional: refresh routing log from API if provided
        const apiBase = import.meta.env.VITE_API_BASE || "";
        if (apiBase) {
          try {
            const r = await fetch(`${apiBase.replace(/\/$/, "")}/routing-log?last_n=100`, { mode: "cors" });
            if (r.ok) {
              const payload = await r.json();
              const routingRows = Array.isArray(payload) ? payload
                : Array.isArray(payload?.entries) ? payload.entries
                : [];
              data.routingRows = routingRows;
            }
          } catch {
            // ignore
          }
        }

        if (!cancelled) {
          setBundle(data);
          setError("");
        }
      } catch (err) {
        if (!cancelled) setError(err.message || "Failed to load dashboard data");
      }
    };

    load();
    const timer = setInterval(load, REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const comparisonRows = bundle?.comparisonRows || [];
  const scatterRows = bundle?.scatterRows || [];
  const latencyRows = bundle?.latencyRows || [];
  const routingRows = bundle?.routingRows || [];
  const generatedAt = bundle?.generatedAt || "";

  // KPIs derived from latencyRows
  const kpis = useMemo(() => {
    if (!latencyRows.length) return null;
    const byUsers = [...latencyRows].sort((a, b) => (a.users || 0) - (b.users || 0));
    const bestThroughput = byUsers.reduce((acc, r) => (r.generate_rps || 0) > (acc?.generate_rps || 0) ? r : acc, null);
    const worstP95 = byUsers.reduce((acc, r) => (r.generate_p95_ms || 0) > (acc?.generate_p95_ms || 0) ? r : acc, null);
    return { bestThroughput, worstP95 };
  }, [latencyRows]);

  const bestThroughput = useMemo(() => {
    if (!comparisonRows.length) return null;
    return [...comparisonRows].sort((a, b) => (b.tok_per_sec || 0) - (a.tok_per_sec || 0))[0];
  }, [comparisonRows]);

  return (
    <main className="container">
      <header className="top">
        <h1>LLM Inference Benchmark Dashboard</h1>
        <p>
          Results from Phase 5 benchmark artifacts and load tests.
          {generatedAt ? ` Last sync: ${generatedAt}` : ""}
        </p>
        {error ? <div className="error">{error}</div> : null}
      </header>

      <div className="grid">
        {kpis ? (
          <Card title="Quick KPIs">
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              <div>
                <strong>Best Throughput</strong>
                <div>{fmt(kpis.bestThroughput?.generate_rps, 2)} req/s at {kpis.bestThroughput?.users} users</div>
              </div>
              <div>
                <strong>Worst P95</strong>
                <div>{fmt(kpis.worstP95?.generate_p95_ms, 0)} ms at {kpis.worstP95?.users} users</div>
              </div>
            </div>
          </Card>
        ) : null}

        <Card
          title="Live Benchmark Comparison Table"
          subtitle={bestThroughput ? `Best throughput: ${bestThroughput.mode}` : ""}
        >
          <table>
            <thead>
              <tr>
                <th>Mode</th>
                <th>Avg Memory (GB)</th>
                <th>Avg TTFT (ms)</th>
                <th>Avg Tok/s</th>
                <th>Avg Quality</th>
              </tr>
            </thead>
            <tbody>
              {comparisonRows.map((row) => (
                <tr key={row.mode}>
                  <td>{row.mode}</td>
                  <td>{fmt(row.mem_gb)}</td>
                  <td>{fmt(row.ttft_ms, 1)}</td>
                  <td>{fmt(row.tok_per_sec, 1)}</td>
                  <td>{row.quality === null ? "-" : fmt(row.quality, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Memory vs Quality Scatter Plot">
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={320}>
              <ScatterChart margin={{ top: 16, right: 16, bottom: 16, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="mem_gb" name="Memory" unit=" GB" />
                <YAxis type="number" dataKey="quality" name="Quality" domain={[0, 5]} />
                <ZAxis type="number" dataKey="tok_per_sec" range={[100, 900]} name="Tok/s" />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Legend />
                <Scatter name="Mode tradeoff" data={scatterRows} fill="#2f80ed" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Latency & Throughput by Users">
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={latencyRows} margin={{ top: 16, right: 16, bottom: 16, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="users" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="generate_avg_ms" name="Avg Latency (ms)" stroke="#27ae60" strokeWidth={2} />
                <Line yAxisId="left" type="monotone" dataKey="generate_p95_ms" name="P95 Latency (ms)" stroke="#eb5757" strokeWidth={2} />
                <Line yAxisId="right" type="monotone" dataKey="generate_rps" name="Throughput (req/s)" stroke="#2f80ed" strokeWidth={2} />
                <Line yAxisId="right" type="monotone" dataKey="generate_fail_ratio" name="Fail Ratio" stroke="#9b51e0" strokeWidth={2} dot />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Routing Decision Log" subtitle={`Showing latest ${routingRows.length} entries`}>
          <div className="log">
            {routingRows.length === 0 ? (
              <p>No routing log entries found.</p>
            ) : (
              routingRows.map((row, idx) => (
                <div className="log-item" key={`${row.timestamp || "t"}-${idx}`}>
                  <span className="badge">{row.routing?.tier || "-"}</span>
                  <div>
                    <div className="log-top">
                      <strong>{row.routing?.precision || "unknown"}</strong>
                      <small>{row.timestamp || ""}</small>
                    </div>
                    <p>{row.prompt || "(no prompt captured)"}</p>
                    <small>
                      tok/s: {fmt(row.metrics?.tok_per_sec)} | ttft_ms: {fmt(row.metrics?.ttft_ms, 1)} | total_ms:{" "}
                      {fmt(row.metrics?.total_ms, 1)}
                    </small>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>
      </div>
    </main>
  );
}

export default App;

