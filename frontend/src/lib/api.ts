import type { AssessmentReport } from "../types/report";

/** `fixture-<name>` loads a synthetic fixture (offline demo and tests); anything else comes from the API. */
export async function loadReport(id: string): Promise<AssessmentReport> {
  const url = id.startsWith("fixture-") ? `/fixtures/${id.slice(8)}.json` : `/api/reports/${id}`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Report not found (${r.status})`);
  return r.json();
}

export const fileUrl = (id: string, name: "brain.nii.gz" | "overlay.nii.gz") =>
  id.startsWith("fixture-") ? null : `/api/reports/${id}/files/${name}${name === "overlay.nii.gz" ? "?view=tech" : ""}`;

export interface FieldSpec {
  name: string; label: string; kind: "number" | "choice"; choices: string[] | null;
  min: number | null; max: number | null; soft_min: number | null; soft_max: number | null;
  default_if_missing: number | string | null; core: boolean; required: boolean;
}
export interface FormSpec {
  fields: FieldSpec[]; required_for_clinical: string[]; modes: { value: string; label: string }[]; max_upload_mb: number;
}
export interface Job {
  id: string; status: "queued" | "running" | "done" | "failed"; step: string | null; mode: string; plan: string[];
  steps_done: string[]; timeline: { step: string; start_s: number }[]; elapsed_s: number; precomputed: boolean;
  error: string | null; report_id: string | null;
}
export interface DemoCase { id: string; label: string; mode_label: string; provenance: string; volume_provider: string }

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = `Request failed (${r.status})`;
    try { const b = await r.json(); if (typeof b.detail === "string") msg = b.detail; } catch { /* keep generic */ }
    throw new Error(msg);
  }
  return r.status === 204 ? (undefined as T) : r.json();
}
export const getSpec = () => j<FormSpec>("/api/form-spec");
export const getDemos = () => j<DemoCase[]>("/api/demo-cases");
export const getJob = (id: string) => j<Job>(`/api/jobs/${id}`);
export const deleteJob = (id: string) => j<void>(`/api/jobs/${id}`, { method: "DELETE" });
export const getModelCard = () => j<{ markdown: string }>("/api/model-card");
export const createJob = (body: FormData) => j<{ job_id: string; warnings: string[] }>("/api/assessments", { method: "POST", body });
