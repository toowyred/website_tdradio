# radio.afest.io routing — how it works

This repo serves `radio.afest.io` via GitHub Pages + a Cloudflare Worker
(`afest-satellites-router`) sitting in front of it. The Worker does path
rewriting + HTML injection at the edge so we never need a folder-per-DJ on
git as the service grows.

## Surface map (as of 2026-04-24)

| URL | Handled by | Purpose |
|---|---|---|
| `radio.afest.io/` | GH Pages | Splash / tune-in animation → redirects to TD's player. |
| `radio.afest.io/v2-beta/` | GH Pages | v2 dev build of the player. Red BETA sticker over the logo. |
| `radio.afest.io/wsi-rx-stable/` | GH Pages | Frozen v1.5 snapshot. Rollback parachute — do NOT touch. |
| `radio.afest.io/u/(name)/` | Worker (rewrite) | Fetches `/v2-beta/` from GH Pages, injects `<base href="https://radio.afest.io/v2-beta/">` + `<script>window.__SATELLITE_USER="(name)";</script>` into `<head>`. Address bar stays at `/u/(name)/`. |
| `radio.afest.io/u/twysted/` | Worker (rewrite) | TD's view — "Director's choice" mode (full channel picker; switching channels doesn't change URL). |
| `radio.afest.io/(djname)` | Worker (302) | Shareable shortcut → `/u/(djname)/`. |
| `radio.afest.io/twysted\|twisted\|twistedduality` | Worker (302) | Alias-map 302 → `/u/twysted/`. Brand hogging — keep forever. |
| `radio.afest.io/wsi-rx` / `/wsi-rx/` | Worker (302) | Escorts to `twysted.afest.io/radio` — no empty-station landing. |
| `twysted.afest.io/radio/u/` | Worker (rewrite) | Phase 1c **test** proxy. Same content as `radio.afest.io/u/twysted/` but with the twysted address bar. Proof-of-concept for the full cut-over. |
| `twysted.afest.io/radio/` | GH Pages (different repo) | Still serves v1.5 prod from `toowyred/webiste_TD`. Untouched until Phase 1c cut-over. |

## Worker source

[`satellites-router.js`](./satellites-router.js) — the whole routing engine is one file. Deploy steps + verification curls are at the top + bottom of that file.

## CF dashboard — required routes

All fail-open, zone: `afest.io`, Worker: `afest-satellites-router`:

1. `radio.afest.io/*`
2. `radio.afest.io/` (CF treats bare root as a separate pattern)
3. `twysted.afest.io/radio/u*` (Phase 1c test — remove when cut-over ships)

## Known Worker gotcha

**Workers do NOT recursively invoke themselves on subrequests.** If the Worker
does `fetch('https://radio.afest.io/u/foo/')` from inside a handler, that
subrequest BYPASSES the Worker's own `/u/(name)/` rule and lands directly on
GH Pages — which has no `/u/foo/` folder → 404.

**Solution**: every path that needs the injection logic calls the same
`serveUserScoped(request, name, rest)` helper inline, which fetches
`${CORE_ORIGIN}${CORE_PATH}` directly (no recursion) and applies the
HTMLRewriter transformation locally.

Don't "let the Worker call itself" — share a helper.

## Future cut-overs (when v2 ships + feature-parity is confirmed)

1. **Phase 1c full** — add Worker routes for `twysted.afest.io/radio` + `/radio/*` that reverse-proxy to `radio.afest.io/u/twysted/*`. Retire the `/radio/u*` test route. Decommission `toowyred/webiste_TD/radio/` GH Pages serve.
2. **Canonical core rename** — move `/v2-beta/` → `/wsi-rx/` in this repo + flip `CORE_PATH` in the Worker. One constant change.
3. **Splash flip** — `radio.afest.io/` currently redirects to `twysted.afest.io/radio/`. Flip to redirect to `radio.afest.io/u/twysted/` once step 1 is live.
4. **Retain** `radio.afest.io/wsi-rx-stable/` for ≥30 days post-cut-over as a rollback parachute, then delete.

## Adding a new DJ (future workflow)

Once the DJ signup service exists:
1. Allocate opaque user ID `usr-XXXX` (4-char base32).
2. Write handle → user-ID mapping to CF KV / D1.
3. Upload their tracks to R2 at `twysted/radio/r/trk-AAAAAAA/*` (opaque track IDs — no DJ name in R2 paths).
4. Write their per-user manifest to `twysted/radio/u/usr-XXXX/tracks.json`.
5. Worker reads handle → user-ID → manifest at request time.

No commit, no deploy, no repo growth.

## Gotchas / tips

- After any CF Worker source edit: **Save & Deploy** — adding a new route doesn't redeploy the source.
- After any CF Worker route add/remove: no deploy needed — routes update instantly.
- Fail-open means if the Worker throws an error or quota is exceeded, requests pass through to GH Pages. Most paths still work; `/u/(name)/` degrades until fixed.
- GH Pages serves static files with `Access-Control-Allow-Origin: *` by default — this is what makes the cross-origin `twysted.afest.io/radio/u/` → radio.afest.io subresource fetches work.
- Subresources (fonts, icons, etc.) resolved via injected `<base>` load directly from `radio.afest.io`. This means `twysted.afest.io/radio/u/*` can't work without radio.afest.io also being live + reachable.
