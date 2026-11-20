import { Box, Chip, fmt, Note, pct, R, STAGES, STAGE_FILL } from "./common";
import type { Driver, StageRange } from "../types/report";

export function StagePanel({ r }: { r: R }) {
  const s = r.stage;
  return (
    <Box id="stage" title="Stage">
      {!s ? <Note>No 4-stage prediction in MRI-only mode.</Note> : (
        <>
          {STAGES.map((k, i) => (
            <div key={k} className={`my-1 grid grid-cols-[48px_1fr_44px] items-center gap-1.5 ${k === s.predicted ? "font-bold" : ""}`}>
              <span>{k}</span>
              <div className="h-4 border border-line bg-soft">
                <div className={`h-full border-r border-black/40 ${STAGE_FILL[i]}`} style={{ width: `${s.model_probability[k] * 100}%` }} />
              </div>
              <span className="text-right">{pct(s.model_probability[k])}</span>
            </div>
          ))}
          <div className="text-xs text-mut">Model probability (frozen system output).</div>
          {s.ambiguity?.flag && (
            <Note warn><b>Uncertain between {s.ambiguity.top2.join(" and ")}</b> (margin {pct(s.ambiguity.margin)}). Treat the stage as approximate.</Note>
          )}
        </>
      )}
    </Box>
  );
}

export function ADPanel({ r }: { r: R }) {
  const src = { KAN_ADD_HEAD: "fusion AD head", MRI_SPECIALIST: "MRI specialist", ALZ_ONLY: "CNN only" }[r.ad.source];
  return (
    <Box id="ad" title="AD probability">
      <div className="flex items-center gap-3">
        <div className="text-3xl font-bold leading-none">{pct(r.ad.model_probability)}</div>
        <div className="text-xs">model probability of AD<br />
          <span className="text-mut">source: {src}{r.ad.alz != null ? ` · CNN score ${r.ad.alz.toFixed(2)}` : ""}</span>
        </div>
      </div>
    </Box>
  );
}

export function SeverityPanel({ r }: { r: R }) {
  const sev = r.stage?.severity;
  return (
    <Box id="severity" title="Severity">
      {!sev ? <p className="text-xs text-mut">Severity not shown{r.meta.mode === "CLINICAL_ONLY" ? ": this is the fusion model run without scan inputs" : ""}.</p> : (
        <>
          <div className="relative h-[18px] border border-line" style={{ background: "linear-gradient(90deg,#dbe9f6,#9ecae1,#4292c6,#08306b)" }}>
            <i className="absolute -top-1 h-6 w-[3px] border border-white bg-black" style={{ left: `calc(${(sev.value / 3) * 100}% - 2px)` }} />
          </div>
          <div className="flex justify-between text-[11px] text-mut"><span>0 CN</span><span>1</span><span>2</span><span>3 AD</span></div>
          <div className="text-xs"><b>Severity {sev.value.toFixed(2)}</b> / 3 · expected stage index, from model probabilities</div>
        </>
      )}
    </Box>
  );
}

export function BrainAgePanel({ r }: { r: R }) {
  const b = r.brain_age;
  return (
    <Box id="brain-age" title="Brain age">
      {b.status === "not_available" ? <p className="text-xs text-mut">Not available.</p> : (
        <div className="flex items-center gap-3">
          <div className="text-3xl font-bold leading-none">{(b.corrected_gap ?? 0) > 0 ? "+" : ""}{(b.corrected_gap ?? 0).toFixed(1)} y</div>
          <div className="text-xs">corrected brain-age gap<br />
            <span className="text-mut">predicted {b.predicted?.toFixed(1)} y at age {b.age_at_scan?.toFixed(1)}{b.status === "unreliable" ? " · UNRELIABLE" : ""}</span>
          </div>
        </div>
      )}
    </Box>
  );
}

export function FlagsPanel({ r }: { r: R }) {
  const f = r.flags ?? [];
  return (
    <Box id="flags" title="Flags">
      {f.length ? f.map((x) => <Note key={x.id}>{x.text}</Note>) : <p className="text-xs text-mut">No flags.</p>}
    </Box>
  );
}

function Band({ d, stage }: { d: Driver; stage: string }) {
  const rg = d.per_stage_ranges as Record<string, StageRange> | null | undefined;
  const v = typeof d.value === "number" ? d.value : null;
  if (!rg || v === null || !rg[stage]) return <span className="text-mut">–</span>;
  const all = Object.values(rg).flatMap((x) => [x.p25, x.p75]).concat(v);
  const lo = Math.min(...all), w = Math.max(...all) - lo || 1;
  const x = (n: number) => ((n - lo) / w) * 100;
  const q = rg[stage];
  return (
    <div className="relative h-2.5 min-w-[110px] border border-line bg-soft" title={`${stage} p25–p75`}>
      <b className="absolute top-0 h-full bg-s1" style={{ left: `${x(q.p25)}%`, width: `${x(q.p75) - x(q.p25)}%` }} />
      <i className="absolute -top-[3px] h-3.5 w-[3px] bg-black" style={{ left: `${x(v)}%` }} />
    </div>
  );
}

function DriverTable({ rows, stage, band }: { rows: Driver[]; stage: string; band: string }) {
  return (
    <table className="t">
      <thead><tr><th>Input</th><th className="n">Value</th><th>{band}</th><th className="n">Push</th></tr></thead>
      <tbody>
        {rows.map((d) => (
          <tr key={d.feature}>
            <td>{d.label}{d.imputed ? " (imputed)" : ""}</td><td className="n">{fmt(d.value)}</td>
            <td><Band d={d} stage={stage} /></td><td className="n">{d.shap > 0 ? "▲" : "▼"} {Math.abs(d.shap).toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function DriversPanel({ r }: { r: R }) {
  const d = r.drivers!;
  const stage = r.stage?.predicted ?? "AD";
  const cStage = d.clinical_target ?? stage;
  const mStage = d.mri_target ?? stage;
  const none = !d.stream_level?.length && !d.clinical?.length && !d.mri?.length;
  return (
    <Box id="drivers" title="What drives this result" panel="drivers">
      {none && <p className="text-xs text-mut">No driver panel for this mode.</p>}
      {!!d.stream_level?.length && (
        <table className="t">
          <thead><tr><th>Stream</th><th className="n">Contribution</th></tr></thead>
          <tbody>{d.stream_level.filter((x) => x.consistent).map((x) => (
            <tr key={x.group}><td>{x.group}</td><td className="n">{x.mean > 0 ? "▲" : "▼"} {Math.abs(x.mean).toFixed(2)}</td></tr>
          ))}</tbody>
        </table>
      )}
      {!!d.clinical?.length && (<>
        <h2 className="mt-2">Clinical inputs, push toward {cStage} {d.clinical_validation && <Chip status={d.clinical_validation.status === "VALIDATED" ? "PASS" : "WARN"}>{d.clinical_validation.status}</Chip>}</h2>
        <DriverTable rows={d.clinical} stage={cStage} band={`Band: ${cStage} p25–p75`} />
      </>)}
      {!!d.mri?.length && (<>
        <h2 className="mt-2">Imaging inputs, push toward {mStage} {d.mri_validation && <Chip status={d.mri_validation.status === "VALIDATED" ? "PASS" : "WARN"}>{d.mri_validation.status}</Chip>}</h2>
        <DriverTable rows={d.mri} stage={mStage} band={`Band: ${mStage} p25–p75`} />
      </>)}
    </Box>
  );
}

export function CaveatsPanel({ r }: { r: R }) {
  return (
    <Box id="caveats" title="Caveats">
      <ul className="m-0 list-disc pl-[18px] text-xs">{(r.caveats ?? []).map((c) => <li key={c.id}>{c.text}</li>)}</ul>
    </Box>
  );
}

export function QCPanel({ r }: { r: R }) {
  return (
    <Box id="quality" title="Input quality">
      <table className="t"><tbody>
        {r.qc.gates?.map((g) => (
          <tr key={g.name}>
            <td>{g.name}{g.note && g.status !== "PASS" ? <div className="text-xs text-mut">{g.note}</div> : null}</td>
            <td className="n">{fmt(g.value)}</td><td><Chip status={g.status} /></td>
          </tr>
        ))}
      </tbody></table>
      <div className="text-xs text-mut">Inputs imputed with training median: {r.completeness.n_clinical_imputed} · MRI {r.completeness.mri_provided ? "provided" : "not provided"}</div>
    </Box>
  );
}
