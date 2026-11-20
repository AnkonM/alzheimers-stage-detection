import DOMPurify from "dompurify";
import { marked } from "marked";
import { useEffect, useState } from "react";
import { getModelCard } from "../lib/api";
import { useTitle } from "../lib/useTitle";

export default function ModelCard() {
  useTitle("Model card");
  const [html, setHtml] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    getModelCard().then((r) => setHtml(DOMPurify.sanitize(marked.parse(r.markdown, { async: false }) as string))).catch((e) => setErr(e.message));
  }, []);
  return (
    <div className="mx-auto max-w-3xl px-4 py-6">
      {err && <p role="alert" className="text-fail">{err}</p>}
      {html === null && !err && <p className="text-mut">Loading…</p>}
      {html && <article className="prose-card" dangerouslySetInnerHTML={{ __html: html }} />}
    </div>
  );
}
