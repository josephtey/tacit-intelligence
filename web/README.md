# LSV Viewer

Small React app to browse the LabSuperVision dataset (videos paired with gold-standard protocols), with a written explainer of how frontier VLMs turn lab video into structured protocols.

## Setup

```bash
cd web
npm install
npm run build-metadata   # generates public/metadata.json from the LSV CSVs
npm run dev              # http://127.0.0.1:5173
```

The app does **not** stream video — local streaming was too slow over the dev server. Instead, each entry shows the absolute path to the video file with a copy button. Open the file directly in `mpv` / `vlc` / your file manager.

Protocol text is inlined into `public/metadata.json` at build time, so no runtime fetch is needed.

## Re-generating metadata

If the LSV CSVs change, re-run:

```bash
npm run build-metadata
```

This rewrites `public/metadata.json` and inlines protocol text into it (small enough — ~230 KB).
