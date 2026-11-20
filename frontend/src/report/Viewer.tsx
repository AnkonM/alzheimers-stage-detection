import { useEffect, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";

const BOX = 180;  // px; the container has a fixed size so the canvas can never change the layout around it

/**
 * Skull-stripped scan in NiiVue, one instance per mounted panel.
 *  - The instance (and its fresh canvas) is created once per base volume and fully disposed on unmount.
 *  - Switching clinician <-> technical never reloads the base volume: only the overlay volume is added or removed.
 *  - The overlay URL is only ever passed in the technical view (see MriPanel).
 * NiiVue sizes its canvas from offsetWidth/offsetHeight * devicePixelRatio and observes the parent. The canvas is therefore
 * absolutely positioned inside a fixed-size box: its CSS size is the box, never its own bitmap size.
 */
export default function Viewer({ brainUrl, overlayUrl, print, onReady, emptyText = "No MRI used" }: {
  emptyText?: string; brainUrl: string | null; overlayUrl?: string | null; print?: boolean; onReady?: () => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const nvRef = useRef<Niivue | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayVol = useRef<unknown>(null);
  const syncing = useRef(false);
  const wanted = useRef<string | null>(null);
  const printRef = useRef(!!print);
  const readyRef = useRef(onReady);
  wanted.current = overlayUrl ?? null;
  printRef.current = !!print;
  readyRef.current = onReady;
  const [baseReady, setBaseReady] = useState(false);
  const [snap, setSnap] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  // A. instance lifecycle: keyed by the base volume only
  useEffect(() => {
    if (!brainUrl || !host.current) { readyRef.current?.(); return; }
    let dead = false, attached = false;
    const canvas = document.createElement("canvas");            // a fresh canvas per instance: never shared, never reused
    canvas.className = "absolute inset-0";
    host.current.appendChild(canvas);
    const nv = new Niivue({ show3Dcrosshair: false, isColorbar: false, backColor: [0.13, 0.13, 0.13, 1], crosshairColor: [0, 0, 0, 0] });
    const dispose = () => {
      try { for (const v of [...nv.volumes]) nv.removeVolume(v); } catch { /* already gone */ }
      try { nv.cleanup(); } catch { /* observers already released */ }
      try { nv.gl?.getExtension("WEBGL_lose_context")?.loseContext(); } catch { /* no context */ }
      canvas.remove();
    };
    nvRef.current = nv; canvasRef.current = canvas; overlayVol.current = null;
    (canvas as unknown as { __nv: Niivue }).__nv = nv;           // read by the browser tests
    setBaseReady(false); setSnap(null); setErr(null);
    (async () => {
      try {
        await nv.attachToCanvas(canvas);
        attached = true;
        if (dead) { dispose(); return; }                        // unmounted while attaching (e.g. React StrictMode)
        await nv.addVolumeFromUrl({ url: brainUrl, colormap: "gray" });
        if (dead) return;
        nv.setSliceType(nv.sliceTypeMultiplanar);
        nv.opts.multiplanarShowRender = 0;                      // SHOW_RENDER.NEVER: three slices, no 3D tile
        nv.drawScene();
        setBaseReady(true);
      } catch (e) {
        if (!dead) { setErr(String(e)); readyRef.current?.(); }
      }
    })();
    return () => {
      dead = true;
      if (attached) dispose(); else canvas.remove();
      nvRef.current = null; canvasRef.current = null; overlayVol.current = null; syncing.current = false;
    };
  }, [brainUrl]);

  // B. overlay: converge to the wanted state (single-flight, so rapid toggling cannot interleave loads)
  useEffect(() => {
    const nv = nvRef.current;
    if (!baseReady || !nv || syncing.current) return;
    syncing.current = true;
    (async () => {
      try {
        for (;;) {
          const want = wanted.current, have = overlayVol.current;
          if (nvRef.current !== nv) return;                     // disposed meanwhile
          if (want && !have) {
            await nv.addVolumeFromUrl({ url: want, colormap: "viridis", opacity: 0.5 });
            overlayVol.current = nv.volumes[nv.volumes.length - 1];
          } else if (!want && have) {
            nv.removeVolume(have as never);
            overlayVol.current = null;
          } else break;
        }
        nv.drawScene();
        if (printRef.current) {                                 // the PDF cannot print a WebGL canvas: freeze it as an image
          await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
          nv.drawScene();
          if (nvRef.current === nv && canvasRef.current) setSnap(canvasRef.current.toDataURL("image/png"));
        }
        readyRef.current?.();
      } catch (e) {
        if (nvRef.current === nv) { setErr(String(e)); readyRef.current?.(); }
      } finally { syncing.current = false; }
    })();
  }, [overlayUrl, baseReady]);

  if (!brainUrl) return <div className="flex h-[130px] items-center justify-center bg-neutral-800 text-xs text-neutral-300">{emptyText}</div>;
  return (
    <div data-viewer>
      {err && <div role="alert" className="text-xs text-fail">Viewer unavailable: {err}</div>}
      <div className="relative w-full overflow-hidden bg-neutral-800" style={{ height: BOX }}>
        <div ref={host} className="absolute inset-0" style={{ visibility: snap ? "hidden" : "visible" }} />
        {snap && <img src={snap} alt="Skull-stripped T1 scan" className="absolute inset-0 h-full w-full bg-neutral-800 object-contain" />}
      </div>
    </div>
  );
}
