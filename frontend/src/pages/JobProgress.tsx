import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import { deleteJob, getJob, Job } from "../lib/api";
import { useTitle } from "../lib/useTitle";

const LABEL: Record<string, string> = {
  validate: "Check inputs", preprocess: "Preprocess the scan", volumes: "Measure brain volumes",
  model: "Run the models and explanations", report: "Assemble the report",
};

export default function JobProgress() {
  useTitle("Job progress");
  const { id = "" } = useParams();
  const nav = useNavigate();
  const warnings: string[] = (useLocation().state as { warnings?: string[] } | null)?.warnings ?? [];
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const j = await getJob(id);
        if (stop) return;
        setJob(j);
        if (j.status === "done") { setTimeout(() => !stop && nav(`/report/${j.report_id}`, { replace: true }), 1200); return; }
        if (j.status === "failed") return;
      } catch (e) { if (!stop) { setErr((e as Error).message); return; } }
      setTimeout(tick, 1000);
    };
    tick();
    return () => { stop = true; };
  }, [id, nav]);

  async function remove() {
    try { await deleteJob(id); nav("/", { replace: true }); } catch (e) { setErr((e as Error).message); }
  }
  if (err) return <div className="mx-auto max-w-3xl px-4 py-6"><p role="alert" className="text-fail">{err}</p><Link className="underline" to="/">Back</Link></div>;
  if (!job) return <div className="mx-auto max-w-3xl px-4 py-6 text-mut">Loading…</div>;

  const start = Object.fromEntries(job.timeline.map((t) => [t.step, t.start_s]));
  const order = job.plan;
  const last = Math.max(-1, ...job.steps_done.map((n) => order.indexOf(n)));
  const status = (_s: string, i: number) => {
    if (job.status === "done") return "done";
    if (i < last) return "done";
    if (i === last) return job.status === "failed" ? "failed" : "running";
    return job.status === "failed" && last === -1 && i === 0 ? "failed" : "pending";
  };
  return (
    <div className="mx-auto max-w-3xl px-4 py-6">
      <h1 className="mb-1 text-lg font-bold">Job progress</h1>
      <p className="mb-3 text-sm text-mut" aria-live="polite">
        {job.status === "done" ? "Done. Opening the report…" : job.status === "failed" ? "Failed." : `${job.status === "queued" ? "Queued" : "Running"} · ${Math.round(job.elapsed_s)} s elapsed`}
        {job.status !== "done" && job.status !== "failed" && " · a new upload takes about 2–3 minutes"}
      </p>
      {warnings.map((w) => <p key={w} className="mb-1 text-xs text-warn">{w}</p>)}
      <ol className="mb-4 divide-y divide-line rounded border border-line bg-white">
        {order.map((s, i) => {
          const st = status(s, i);
          return (
            <li key={s} className="flex items-center gap-3 px-3 py-2 text-sm" data-step={s} data-state={st}>
              <span aria-hidden className={`inline-block h-3 w-3 rounded-full ${st === "done" ? "bg-pass" : st === "running" ? "animate-pulse bg-s2" : st === "failed" ? "bg-fail" : "bg-line"}`} />
              <span className="grow">{LABEL[s] ?? s}</span>
              <span className="text-xs text-mut">{st === "done" ? "done" : st === "running" ? "running" : st === "failed" ? "failed" : "waiting"}{s in start ? ` · started at ${start[s]} s` : ""}</span>
            </li>
          );
        })}
      </ol>
      {job.status === "failed" && <p role="alert" className="mb-3 rounded border-l-[3px] border-fail bg-[#fdecea] px-2 py-1.5 text-sm">{job.error ?? "The job failed."}</p>}
      <div className="flex gap-3 text-sm">
        {job.status === "done" && <Link className="rounded bg-s3 px-3 py-1 font-semibold text-white" to={`/report/${job.report_id}`}>Open report</Link>}
        {!job.precomputed && (job.status === "done" || job.status === "failed") &&
          <button className="rounded border border-fail px-3 py-1 text-fail" onClick={remove}>Delete job and its files</button>}
        <Link className="underline" to="/">New assessment</Link>
      </div>
    </div>
  );
}
