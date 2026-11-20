import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { createJob, DemoCase, FieldSpec, FormSpec, getDemos, getSpec } from "../lib/api";
import { useTitle } from "../lib/useTitle";

const input = "w-full rounded border border-line bg-white px-2 py-1 text-sm focus:outline focus:outline-2 focus:outline-s2";

function Field({ f, value, onChange }: { f: FieldSpec; value: string; onChange: (v: string) => void }) {
  const num = Number(value);
  const hard = value !== "" && f.kind === "number" && (Number.isNaN(num) || (f.min != null && num < f.min) || (f.max != null && num > f.max));
  const soft = !hard && value !== "" && f.kind === "number" && f.soft_min != null && f.soft_max != null && (num < f.soft_min || num > f.soft_max);
  const id = `f-${f.name}`, hint = `${id}-hint`;
  return (
    <div className="flex flex-col gap-0.5">
      <label htmlFor={id} className="text-xs font-semibold">{f.label}{f.required && <span aria-hidden> *</span>}</label>
      {f.kind === "choice" ? (
        <select id={id} className={input} value={value} onChange={(e) => onChange(e.target.value)} aria-describedby={hint} required={f.required}>
          <option value="">{f.required ? "Select…" : `Not provided (training value: ${f.default_if_missing})`}</option>
          {f.choices!.map((c) => <option key={c}>{c}</option>)}
        </select>
      ) : (
        <input id={id} className={input} inputMode="decimal" value={value} onChange={(e) => onChange(e.target.value)}
          aria-invalid={hard} aria-describedby={hint} required={f.required} placeholder={f.min != null ? `${f.min}–${f.max}` : ""} />
      )}
      <div id={hint} className="min-h-4 text-[11px]">
        {hard && <span className="text-fail">Must be a number from {f.min} to {f.max}.</span>}
        {soft && <span className="text-warn">Outside the training subjects ({f.soft_min}–{f.soft_max}): the result is an extrapolation.</span>}
        {!hard && !soft && value === "" && !f.required && <span className="text-mut">Imputed with the training median ({String(f.default_if_missing)}).</span>}
      </div>
    </div>
  );
}

export default function NewAssessment() {
  useTitle("New assessment");
  const nav = useNavigate();
  const [spec, setSpec] = useState<FormSpec | null>(null);
  const [demos, setDemos] = useState<DemoCase[]>([]);
  const [mode, setMode] = useState("FULL");
  const [vals, setVals] = useState<Record<string, string>>({});
  const [file, setFile] = useState<File | null>(null);
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [showMore, setShowMore] = useState(false);

  useEffect(() => { getSpec().then(setSpec).catch((e) => setErr(String(e.message))); getDemos().then(setDemos).catch(() => {}); }, []);
  const needScan = mode !== "CLINICAL_ONLY", needClin = mode !== "MRI_ONLY";
  const fields = useMemo(() => (spec?.fields ?? []).filter((f) => (mode === "MRI_ONLY" ? f.name === "AGE" || f.name === "PTGENDER" : true)), [spec, mode]);
  const core = fields.filter((f) => f.core), more = fields.filter((f) => !f.core);
  const imputed = needClin ? (spec?.fields ?? []).filter((f) => !f.required && (vals[f.name] ?? "") === "").map((f) => f.label) : [];
  const tooBig = !!file && !!spec && file.size > spec.max_upload_mb * 1048576;
  const badType = !!file && !/\.nii(\.gz)?$/i.test(file.name);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setErr(null);
    if (needScan && !file) return setErr("Choose a T1 scan (NIfTI).");
    if (tooBig || badType) return setErr("The scan must be a NIfTI file within the size limit.");
    const fd = new FormData();
    fd.set("mode", mode);
    if (label) fd.set("subject_label", label);
    if (needClin || mode === "MRI_ONLY") {
      const out: Record<string, string | number> = {};
      for (const f of fields) {
        const v = (vals[f.name] ?? "").trim();
        if (v !== "") out[f.name] = f.kind === "number" ? Number(v) : v;
      }
      fd.set("clinical_json", JSON.stringify(out));
    }
    if (needScan && file) fd.set("scan", file);
    setBusy(true);
    try { const r = await createJob(fd); nav(`/jobs/${r.job_id}`, { state: { warnings: r.warnings } }); }
    catch (x) { setErr((x as Error).message); setBusy(false); }
  }

  return (
    <div className="mx-auto grid max-w-5xl gap-6 px-4 py-6 md:grid-cols-[1fr_16rem]">
      <form onSubmit={submit} className="rounded border border-line bg-white p-4" aria-labelledby="h-new" noValidate>
        <h1 id="h-new" className="mb-3 text-lg font-bold">New assessment</h1>
        <fieldset className="mb-4">
          <legend className="mb-1 text-sm font-semibold">Mode</legend>
          <div className="flex flex-col gap-1">
            {(spec?.modes ?? []).map((m) => (
              <label key={m.value} className="flex items-center gap-2 text-sm">
                <input type="radio" name="mode" value={m.value} checked={mode === m.value} onChange={() => setMode(m.value)} /> {m.label}
              </label>
            ))}
          </div>
        </fieldset>
        {needScan && (
          <div className="mb-4">
            <label htmlFor="scan" className="text-sm font-semibold">T1 MRI (NIfTI)</label>
            <input id="scan" type="file" accept=".nii,.nii.gz" className={input} aria-describedby="scan-hint"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            <p id="scan-hint" className="text-xs text-mut">
              .nii or .nii.gz only, up to {spec?.max_upload_mb ?? "…"} MB. DICOM is not supported. The upload is deleted after processing.
            </p>
            {tooBig && <p role="alert" className="text-xs text-fail">The file is larger than the limit.</p>}
            {badType && <p role="alert" className="text-xs text-fail">Only NIfTI files are accepted.</p>}
          </div>
        )}
        <div className="mb-4">
          <label htmlFor="label" className="text-sm font-semibold">Label (optional)</label>
          <input id="label" className={input} maxLength={40} value={label} onChange={(e) => setLabel(e.target.value)} aria-describedby="label-hint" />
          <p id="label-hint" className="text-xs text-mut">A pseudonymous name for this report. Do not enter a subject ID.</p>
        </div>
        {fields.length > 0 && (
          <fieldset className="mb-3">
            <legend className="mb-1 text-sm font-semibold">{needClin ? "Clinical inputs" : "Age and sex"} <span className="font-normal text-mut">(* required)</span></legend>
            <div className="grid gap-x-3 gap-y-1 sm:grid-cols-2">
              {core.map((f) => <Field key={f.name} f={f} value={vals[f.name] ?? ""} onChange={(v) => setVals({ ...vals, [f.name]: v })} />)}
            </div>
            {more.length > 0 && (
              <>
                <button type="button" className="mt-2 text-sm text-s3 underline" aria-expanded={showMore} onClick={() => setShowMore(!showMore)}>
                  {showMore ? "Hide" : "Show"} more optional fields ({more.length})
                </button>
                {showMore && <div className="mt-2 grid gap-x-3 gap-y-1 sm:grid-cols-2">
                  {more.map((f) => <Field key={f.name} f={f} value={vals[f.name] ?? ""} onChange={(v) => setVals({ ...vals, [f.name]: v })} />)}
                </div>}
              </>
            )}
            {needClin && imputed.length > 0 && (
              <p className="mt-3 rounded border-l-[3px] border-s2 bg-soft px-2 py-1.5 text-xs" aria-live="polite">
                {imputed.length} fields left empty will be imputed with the training median and listed in the report: {imputed.join(", ")}.
              </p>
            )}
          </fieldset>
        )}
        {err && <p role="alert" className="mb-2 text-sm text-fail">{err}</p>}
        <button type="submit" disabled={busy || !spec} className="rounded bg-s3 px-4 py-1.5 text-sm font-semibold text-white disabled:opacity-50">
          {busy ? "Uploading…" : "Run assessment"}
        </button>
        <p className="mt-2 text-xs text-mut">A new upload takes about 2–3 minutes.</p>
      </form>
      <aside aria-labelledby="h-demo" className="rounded border border-line bg-white p-4 text-sm">
        <h2 id="h-demo" className="mb-2 font-semibold">Load demo case</h2>
        <p className="mb-2 text-xs text-mut">Precomputed reports open instantly. Cases come from ADNI and were in the system's training data.</p>
        {demos.length === 0 ? <p className="text-xs text-mut">No demo cases found.</p> : (
          <ul className="flex flex-col gap-2">
            {demos.map((d) => (
              <li key={d.id}>
                <Link className="font-semibold text-s3 underline" to={`/report/${d.id}`}>{d.label}</Link>
                <div className="text-xs text-mut">{d.mode_label}{d.provenance === "new_upload" ? " · upload path demonstration" : ""}</div>
              </li>
            ))}
          </ul>
        )}
      </aside>
    </div>
  );
}
