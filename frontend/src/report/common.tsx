import type { ReactNode } from "react";
import type { AssessmentReport } from "../types/report";

export const STAGES = ["CN", "EMCI", "LMCI", "AD"] as const;
export const STAGE_FILL = ["bg-s0", "bg-s1", "bg-s2", "bg-s3"];
export const pct = (x: number) => `${Math.round(x * 100)}%`;
export type View = "clin" | "tech";
export type R = AssessmentReport;

export function Chip({ status, children }: { status: "PASS" | "WARN" | "FAIL" | string; children?: ReactNode }) {
  const c = status === "PASS" || status === "VALIDATED" ? "text-pass" : status === "FAIL" ? "text-fail" : "text-warn";
  return <span className={`inline-block rounded-full border px-2 text-xs font-semibold ${c} border-current`}>{children ?? status}</span>;
}

/** Sections are numbered by a CSS counter in document order, so a mode that omits a section stays 1..n without gaps. */
export function Box({ id, title, children, panel }: { id: string; title: string; children: ReactNode; panel?: string }) {
  return (
    <section className="box" data-section={id}>
      <h2 className="sec-title">{title}</h2>
      <div data-panel={panel}>{children}</div>
    </section>
  );
}

export const Note = ({ children, warn }: { children: ReactNode; warn?: boolean }) => (
  <p className={`my-1 border-l-[3px] px-2 py-1.5 text-[12.5px] ${warn ? "border-warn bg-[#fff8e6]" : "border-s2 bg-soft"}`}>{children}</p>
);

/** Display formatting only: values arrive unrounded from the engine and are rounded here. */
export function fmt(v: unknown): string {
  if (v === null || v === undefined) return "–";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(3)));
  return String(v);
}
