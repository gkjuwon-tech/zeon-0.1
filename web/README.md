# web/

The ZEON landing page. Static HTML + CSS, no build step, no JS framework, no
analytics. Open `index.html` in a browser:

```bash
# from the repo root:
python -m http.server -d web 8765
open http://localhost:8765/
```

Design notes:
- HN-style orange bar on top, terminal-style body underneath.
- Single CSS file. No external fonts (we use the OS monospace stack on
  purpose).
- All link targets are absolute GitHub URLs so the page also works fine
  if it's served from a static host (Pages, Vercel, anywhere).

If you want to ship a new section, just edit `index.html`. No fancy
templating; if you find yourself adding a build tool, stop.
