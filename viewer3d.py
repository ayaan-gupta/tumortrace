"""Client-side interactive 3D brain + tumor-region viewer.

Built on NiiVue (https://niivue.github.io — the same WebGL viewer used in
real neuroimaging research tooling), embedded via an inline HTML component.
Rendering happens entirely in the browser via WebGL: there is no server-side
VTK/OSMesa/xvfb dependency, which is what makes this practical to deploy on
Streamlit Community Cloud (a native Python 3D-rendering library like PyVista
would need a headless-GL setup that free container hosts don't reliably
provide).

The base volume and predicted mask are embedded directly as base64-encoded
NIfTI bytes in the generated HTML (no external file server involved), so
this works identically whether the volume came from a bundled sample or a
user upload.
"""
import os
import tempfile
from base64 import b64encode

import nibabel as nib
import numpy as np

NIIVUE_CDN_URL = "https://cdn.jsdelivr.net/npm/@niivue/niivue@0.69.0/dist/niivue.umd.js"


def _nifti_base64(volume, affine, dtype):
    """Round-trips through a real temp file (nibabel needs a path-like target
    to write compressed .nii.gz correctly) -- same pattern app.py already
    uses for the mask download button."""
    tmp_path = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    try:
        nib.save(nib.Nifti1Image(volume.astype(dtype), affine), tmp_path)
        with open(tmp_path, "rb") as f:
            return b64encode(f.read()).decode()
    finally:
        os.remove(tmp_path)


def encode_volumes(base_volume, pred_mask, affine):
    """base_volume: single-modality (H,W,D) float array. pred_mask: (H,W,D)
    uint8 label volume. Returns (base_b64, mask_b64) ready to embed."""
    base_b64 = _nifti_base64(base_volume, affine, dtype=np.float32)
    mask_b64 = _nifti_base64(pred_mask, affine, dtype=np.uint8)
    return base_b64, mask_b64


def build_viewer_html(base_b64, mask_b64, height=560, base_opacity=0.55, tumor_opacity=0.9):
    """Returns a full standalone HTML document string for use as an iframe
    srcdoc (via st.components.v1.html). Includes its own cross-section-depth
    and tumor-opacity sliders since a component iframe can't be wired to
    external Streamlit widgets without a custom bidirectional component."""
    canvas_height = max(height - 90, 240)
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  html, body {{ margin:0; background:#0b0f14; color:#7d8b99; font-family:'IBM Plex Mono',monospace; font-size:11px; }}
  #status {{ padding:5px 8px; letter-spacing:0.04em; text-transform:uppercase; font-size:10px; }}
  /* NiiVue sets the canvas's own inline style to width:100%/height:100%, which needs
     a definite-height ancestor to resolve -- without this wrapper the canvas collapses
     to the browser's built-in 150px default and the whole render looks tiny. */
  #gl-wrap {{ width:100%; height:{canvas_height}px; }}
  #gl {{ display:block; }}
  #controls {{ padding:6px 8px; display:flex; gap:1.4rem; align-items:center; flex-wrap:wrap; border-top:1px solid #263241; }}
  input[type=range] {{ accent-color:#ff5a45; }}
  button {{ background:transparent; border:1px solid #263241; color:#ff5a45; font-family:inherit;
            font-size:10px; letter-spacing:0.04em; text-transform:uppercase; padding:4px 10px;
            border-radius:2px; cursor:pointer; }}
  button:hover {{ background:#ff5a45; color:#0b0f14; }}
</style></head>
<body>
<div id="status">loading 3D viewer…</div>
<div id="gl-wrap"><canvas id="gl"></canvas></div>
<div id="controls">
  <label>Cross-section depth <input type="range" id="clip" min="0" max="100" value="100"></label>
  <label>Tumor opacity <input type="range" id="op" min="0" max="100" value="{int(tumor_opacity * 100)}"></label>
  <button id="resetBtn">Reset view</button>
</div>
<script src="{NIIVUE_CDN_URL}"></script>
<script>
  const statusEl = document.getElementById('status');
  (async () => {{
    try {{
      const {{ Niivue }} = window.niivue;
      const nv = new Niivue({{ show3Dcrosshair: false, backColor: [0.043, 0.059, 0.078, 1] }});
      await nv.attachToCanvas(document.getElementById('gl'));
      await nv.loadVolumes([
        {{ url: "data:application/gzip;base64,{base_b64}", name: "base.nii.gz", colormap: "gray", opacity: {base_opacity} }},
        {{ url: "data:application/gzip;base64,{mask_b64}", name: "mask.nii.gz", colormap: "warm", opacity: {tumor_opacity} }}
      ]);
      nv.setSliceType(nv.sliceTypeRender);
      // NiiVue renders the clip plane itself as a solid magenta slab by default
      // (opts.clipPlaneColor = [0.7, 0, 0.7, 0.5]) -- that's the "weird purple plane".
      // The clipping (the actual geometry cut) is separate from that visual and stays
      // intact; setClipPlaneColor (not a direct opts mutation, which doesn't reach the
      // renderer) makes the plane indicator fully transparent so only the cut brain
      // surface shows.
      nv.setClipPlaneColor([1, 1, 1, 0]);
      // Default 3D render scale (1.0) leaves a lot of empty canvas around the brain.
      nv.volScaleMultiplier = 1.8;
      // Slider starts at 100 ("no clip"), so the initial clip state must match that,
      // not a partially-clipped view -- otherwise the control and the render disagree
      // the moment the viewer loads.
      nv.setClipPlane([0, 270, 0]);
      nv.setRenderAzimuthElevation(110, 10);
      // Force a resize pass now that the canvas sits in a definite-height wrapper --
      // niivue only recomputes its GL buffer size in response to a resize event.
      nv.resizeListener();
      nv.drawScene();

      document.getElementById('clip').addEventListener('input', (e) => {{
        nv.setClipPlane([(1 - e.target.value / 100), 270, 0]);
      }});
      document.getElementById('op').addEventListener('input', (e) => {{
        nv.setOpacity(1, e.target.value / 100);
      }});
      document.getElementById('resetBtn').addEventListener('click', () => {{
        document.getElementById('clip').value = 100;
        nv.setClipPlane([0, 270, 0]);
        nv.volScaleMultiplier = 1.8;
        nv.setRenderAzimuthElevation(110, 10);
      }});

      statusEl.textContent = "drag to rotate · scroll to zoom · sliders below for cross-section + opacity";
    }} catch (err) {{
      statusEl.textContent = "3D viewer failed to load: " + err.message;
      console.error(err);
    }}
  }})();
</script>
</body></html>"""
