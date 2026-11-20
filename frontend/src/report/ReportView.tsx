import { useEffect, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { loadReport } from "../lib/api";
import { Chip, R, View } from "./common";
import { useTitle } from "../lib/useTitle";
import { ADPanel, BrainAgePanel, CaveatsPanel, DriversPanel, FlagsPanel, QCPanel, SeverityPanel, StagePanel } from "./Panels";
import MriPanel from "./MriPanel";

declare global { interface Window { __reportReady?: boolean } }

export default function ReportView() {
  const { id = "" } = useParams();
  const [q, setQ] = useSearchParams();
  const print = q.get("print") === "1";
  const view: View = q.get("view") === "tech" ? "tech" : "clin";
  const [r, setR] = useState<R | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [viewerReady, setViewerReady] = useState(false);
  useTitle("Report");

  useEffect(() => { loadReport(id).then(setR).catch((e) => setErr(String(e))); }, [id]);
  useEffect(() => { window.__reportReady = !!r && (viewerReady || r.meta.mode === "CLINICAL_ONLY"); }, [r, viewerReady]);

  if (err) return <div className="sheet text-fail">{err}</div>;
  if (!r) return <div className="sheet text-mut">Loading report…</div>;
  const m = r.meta;
  const noScan = m.mode === "CLINICAL_ONLY";            // no MRI panel and no brain age in this mode
  const set = (v: View) => { const n = new URLSearchParams(q); n.set("view", v); setQ(n, { replace: true }); };

  return (
    <>
      {!print && (
        <div className="no-print mx-auto flex max-w-[210mm] items-center justify-end gap-2 px-4 pt-3 text-sm">
          <div role="group" aria-label="Report view" className="inline-flex overflow-hidden rounded border border-line bg-white">
            {(["clin", "tech"] as const).map((v) => (
              <button key={v} aria-pressed={view === v} onClick={() => set(v)}
                className={`px-3 py-1 ${view === v ? "bg-s3 text-white" : "hover:bg-soft"}`}>{v === "clin" ? "Clinician" : "Technical"}</button>
            ))}
          </div>
          {id.startsWith("fixture-") === false && (
            <a className="rounded border border-line bg-white px-3 py-1 hover:bg-soft" href={`/api/reports/${id}.pdf?view=${view}`} download>Download PDF ({view === "clin" ? "clinician" : "technical"} view)</a>
          )}
        </div>
      )}
      <main className="sheet" data-report-ready={viewerReady ? "1" : "0"}>
        <div className="flex items-center gap-3">
          <div className="grow">
            <h1 className="m-0 text-xl font-bold">Cognitive assessment report</h1>
            <div className="text-xs text-mut">
              {m.subject_label} · {m.mode_label}{m.requested_mode !== m.mode ? ` (requested ${m.requested_mode.toLowerCase()})` : ""} ·{" "}
              {m.input_provenance === "adni_demo" ? "ADNI demo case" : "new upload"} · {m.created_at.slice(0, 10)} · {m.system_version}
            </div>
          </div>
          <Chip status={r.qc.overall} />
        </div>
        <div className="cols2 mt-2 grid grid-cols-2 gap-x-4">
          <div>
            <StagePanel r={r} /><ADPanel r={r} /><SeverityPanel r={r} />
            {!noScan && <BrainAgePanel r={r} />}
            <FlagsPanel r={r} />
          </div>
          <div>
            {!noScan && <MriPanel r={r} id={id} view={view} print={print} onReady={() => setViewerReady(true)} />}
            <QCPanel r={r} />
          </div>
        </div>
        <DriversPanel r={r} />
        <CaveatsPanel r={r} />
        <div className="mt-2 border-t border-line pt-1.5 text-[11px] text-mut">
          <b>{r.research_use_notice}</b> Reference population: {m.reference_population.description}, n={m.reference_population.n}.
          System {m.system_version}. Report {m.report_id}.
        </div>
      </main>
    </>
  );
}
