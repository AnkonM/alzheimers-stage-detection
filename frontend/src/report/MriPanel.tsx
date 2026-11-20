import { Box, Chip, fmt, Note, R, View } from "./common";
import Viewer from "./Viewer";
import { fileUrl } from "../lib/api";

function Volumes({ r }: { r: R }) {
  const v = r.volumes ?? [];
  if (!v.length) return <p className="text-xs text-mut">Not assessed (no usable scan).</p>;
  const synth = r.meta.volume_provider === "synthseg";
  return (
    <div data-panel="volumes">
      <table className="t">
        <thead><tr><th>Structure</th><th className="n">{synth ? "Ratio to ICV" : "% of CN median"}</th><th className="n">Percentile</th></tr></thead>
        <tbody>
          {v.map((x) => (
            <tr key={x.structure}>
              <td>{x.structure}</td>
              <td className="n">{!synth && x.pct_of_cn_median != null ? `${Math.round(x.pct_of_cn_median)}%` : fmt(x.ratio_to_icv)}</td>
              <td className="n">{x.percentile != null ? Math.round(x.percentile) : <span className="text-xs text-mut">{x.display_note ?? "–"}</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="text-xs text-mut">
        Percentile among CN training subjects (n={r.meta.reference_population.n}); {r.meta.reference_population.volume_percentiles}.
      </div>
    </div>
  );
}

function CNN({ r }: { r: R }) {
  const c = r.cnn;
  if (!c) return null;
  const w = c.wording[0].toUpperCase() + c.wording.slice(1);
  return (
    <div data-panel="cnn" className="mt-2">
      <h2>CNN attribution (technical)</h2>
      <p className="my-1 flex gap-1.5"><Chip status="FAIL">Cascade test FAIL</Chip><Chip status="FAIL">Deletion test FAIL</Chip></p>
      <Note warn>{w}. The map failed both pre-registered sanity checks; it does not show where disease is.</Note>
      <table className="t cnn-table">
        <thead><tr><th>Atlas region</th><th className="n">Density</th><th className="n">AD − CN</th></tr></thead>
        <tbody>
          {c.regions.map((x) => (
            <tr key={x.region}><td>{x.region}</td><td className="n">{x.density.toFixed(3)}</td>
              <td className="n">{(x.ad_minus_cn ?? 0) >= 0 ? "+" : ""}{(x.ad_minus_cn ?? 0).toFixed(3)}</td></tr>
          ))}
        </tbody>
      </table>
      <div className="text-xs text-mut">Fixed atlas order; not sorted. {c.layer}, {c.n_models} models.</div>
    </div>
  );
}

export default function MriPanel({ r, id, view, print, onReady }: { r: R; id: string; view: View; print: boolean; onReady: () => void }) {
  const brain = r.meta.has_brain_image ? fileUrl(id, "brain.nii.gz") : null;
  const overlay = view === "tech" && r.cnn?.overlay_file ? fileUrl(id, "overlay.nii.gz") : null;
  return (
    <Box id="mri" title="MRI">
      <Viewer brainUrl={brain} overlayUrl={overlay} print={print} onReady={onReady}
        emptyText={r.meta.has_brain_image ? "Scan viewer needs a live report (fixtures have no image)" : "No MRI used"} />
      {r.meta.volume_provider === "synthseg" && (
        <Note warn>Volumes come from SynthSeg, not the FreeSurfer volumes the model was trained on. Where agreement was low, no percentile is shown. See Input quality.</Note>
      )}
      <div className="mt-1.5"><Volumes r={r} /></div>
      {view === "tech" && <CNN r={r} />}
    </Box>
  );
}
