/**
 * afest-satellites-router — multi-tenant routing Worker for radio.afest.io
 * ────────────────────────────────────────────────────────────────────────────
 * Sibling to the existing `afest-assets-gate` Worker on assets.afest.io. This
 * one owns the public-facing radio.afest.io/* surface PLUS the Phase 1c test
 * proxy at twysted.afest.io/radio/u*.
 *
 * radio.afest.io/*
 *   /                          → splash HTML on GH Pages (unchanged)
 *   /v2-beta/*                 → pass-through to GH Pages (v2 dev build)
 *   /wsi-rx/*                  → pass-through to GH Pages (future canonical core)
 *   /wsi-rx (bare) + /wsi-rx/  → 302 to twysted.afest.io/radio (no empty station)
 *   /wsi-rx-stable/*           → pass-through (v1.5 rollback parachute)
 *   /u/(name)/*                → edge-rewrite: fetch /v2-beta/<rest>, inject
 *                                 <base href="https://radio.afest.io/v2-beta/">
 *                                 + a one-liner script setting
 *                                 window.__SATELLITE_USER=(name). Address bar
 *                                 stays at /u/(name)/. Base is absolute so the
 *                                 same injection works when the HTML is
 *                                 reverse-proxied cross-origin (see below).
 *   /(name) (bare top segment) → 302 → /u/(name)/  (shareable shortcut)
 *   /twysted, /twisted, /twistedduality → 302 → /u/twysted (alias map)
 *
 * twysted.afest.io/radio/u*                 [Phase 1c test proxy]
 *   Reverse-proxies to radio.afest.io/u/twysted/* while preserving the
 *   twysted.afest.io/radio/u address bar. Proof-of-concept for the eventual
 *   Phase 1c cut-over where twysted.afest.io/radio/ itself becomes a proxy
 *   shell over /u/twysted/. Cross-origin subresources work because GH Pages
 *   serves radio.afest.io with `Access-Control-Allow-Origin: *`.
 *
 * Any reserved path segment (the static folder names listed in RESERVED) is
 * NEVER treated as a DJ handle — it always passes through to the origin.
 *
 * When the canonical player later moves from /v2-beta/ to /wsi-rx/, flip the
 * single CORE_PATH constant below. No other changes required.
 *
 * ─ Deploy ────────────────────────────────────────────────────────────────
 * Cloudflare Dashboard → Workers & Pages → `afest-satellites-router` → paste
 * this file → Save & Deploy. Required Routes (Zone: afest.io):
 *   1. radio.afest.io/*            (covers /u/*, /v2-beta/*, etc.)
 *   2. radio.afest.io/             (bare root — separate pattern in CF)
 *   3. twysted.afest.io/radio/u    (the Phase 1c test — bare)
 *   4. twysted.afest.io/radio/u/*  (the Phase 1c test — deeper paths)
 * Use fail-open on all routes; this Worker is routing-only, no auth.
 *
 * Rollback: delete the route(s) in the CF dashboard. All paths fall back to
 * GH Pages raw serving. `/u/(name)/` URLs stop working after rollback.
 */

// ── Config ───────────────────────────────────────────────────────────────
// Path of the canonical player core on GH Pages. /u/(name)/ requests are
// rewritten to this path at the edge. Flip to '/wsi-rx/' once the player
// moves there (Phase 1c).
const CORE_PATH = '/v2-beta/';

// Where empty-station traffic gets escorted when someone lands on /wsi-rx/
// with no user scope attached.
const ESCORT_URL = 'https://twysted.afest.io/radio';

// Path segments that MUST NOT be treated as DJ handles. Anything here passes
// through to GH Pages verbatim. Grow this list whenever a new top-level
// static folder or file is added to the radio.afest.io repo.
const RESERVED = new Set([
  '',                        // already handled by root rule
  'u',                       // the /u/ prefix itself
  'v2-beta',
  'wsi-rx',
  'wsi-rx-stable',
  'fonts',
  'assets',
  'static',
  'images',
  'favicon.ico',
  'robots.txt',
  'sitemap.xml',
  'cname',
  'index.html',
  'cloudflare-worker.js',
  'satellites-router.js',
  'readme.md',
]);

// Shortcut aliases that resolve to a canonical DJ handle. Both the top-level
// form (`/twisted` → 302 → `/u/twysted/`) and the /u/ form (`/u/twisted/` →
// rewritten as if it were `/u/twysted/`) respect this map.
const HANDLE_ALIAS = {
  twisted: 'twysted',
  twistedduality: 'twysted',
};

// Absolute origin of the canonical core (used for <base> injection so the
// same-origin AND cross-origin proxy forms both resolve subresources correctly).
const CORE_ORIGIN = 'https://radio.afest.io';

// ── Entry point ──────────────────────────────────────────────────────────
export default {
  async fetch(request) {
    const url = new URL(request.url);
    const path = url.pathname;
    const host = url.hostname;

    // ── Phase 1c test proxy on twysted.afest.io/radio/u* ───────────────
    // Reverse-proxy to the core player at radio.afest.io/v2-beta/* with the
    // same <base> + __SATELLITE_USER injection the /u/(name)/ rule uses.
    // IMPORTANT: we fetch /v2-beta/* directly (NOT /u/twysted/*) because
    // Cloudflare Workers don't recursively invoke themselves on
    // subrequests — so a fetch() to radio.afest.io/u/twysted/ from inside
    // this Worker would bypass the /u/(name)/ injection rule and land on
    // GH Pages' non-existent /u/twysted/ folder. Fetching /v2-beta/*
    // directly is the same content the /u/(name)/ rule would produce;
    // serveUserScoped() handles the injection locally.
    if (host === 'twysted.afest.io'
        && (path === '/radio/u' || path === '/radio/u/' || path.startsWith('/radio/u/'))) {
      if (path === '/radio/u/_ping') {
        return new Response('satellites-router alive · twysted proxy route OK', {
          status: 200,
          headers: { 'content-type': 'text/plain; charset=utf-8', 'x-worker': 'satellites-router' },
        });
      }
      const rest = path === '/radio/u' || path === '/radio/u/'
        ? '/'
        : path.slice('/radio/u'.length);
      return serveUserScoped(request, 'twysted', rest);
    }

    // 1) Root → splash, pass through unchanged.
    if (path === '/' || path === '/index.html') {
      return fetch(request);
    }

    // 2) /wsi-rx (bare or trailing-slash-only) → escort to TD's radio. We
    //    never want a user to land on an empty station with no user scope.
    if (path === '/wsi-rx' || path === '/wsi-rx/') {
      return Response.redirect(ESCORT_URL, 302);
    }

    // 3) Known static prefixes → pass through verbatim.
    if (path.startsWith(CORE_PATH) ||
        path.startsWith('/v2-beta/') ||
        path.startsWith('/wsi-rx/') ||
        path.startsWith('/wsi-rx-stable/') ||
        path.startsWith('/fonts/') ||
        path.startsWith('/assets/') ||
        path.startsWith('/static/')) {
      return fetch(request);
    }

    // 4) /u/(name)/* → rewrite to CORE_PATH with <base> + __SATELLITE_USER.
    const uMatch = path.match(/^\/u\/([^\/]+)(\/.*)?$/);
    if (uMatch) {
      const rawName = decodeURIComponent(uMatch[1]);
      const canonicalName = resolveAlias(rawName);
      // If alias normalization changed the handle, redirect so the address
      // bar ends up at the canonical form (SEO + sharing cleanliness).
      if (canonicalName !== rawName) {
        const rest = uMatch[2] || '/';
        return Response.redirect(
          `${url.origin}/u/${encodeURIComponent(canonicalName)}${rest}${url.search}`,
          302
        );
      }
      return serveUserScoped(request, canonicalName, uMatch[2] || '/');
    }

    // 5) Top-level single-segment path → 302 to /u/(name)/ for shareable
    //    shortcuts (radio.afest.io/djsickmode → /u/djsickmode/).
    const segments = path.split('/').filter(Boolean);
    if (segments.length === 1) {
      const raw = decodeURIComponent(segments[0]);
      const rawLower = raw.toLowerCase();
      // Reserved → pass through (static file/folder).
      if (RESERVED.has(rawLower)) return fetch(request);
      const canonical = resolveAlias(raw);
      return Response.redirect(
        `${url.origin}/u/${encodeURIComponent(canonical)}/${url.search}`,
        302
      );
    }

    // 6) Fallback: pass through to GH Pages.
    return fetch(request);
  },
};

// ── Helpers ──────────────────────────────────────────────────────────────
function resolveAlias(handle) {
  return HANDLE_ALIAS[handle.toLowerCase()] || handle;
}

/**
 * Internal fetch the core player HTML/assets, then for HTML responses rewrite
 * the <head> to:
 *   a) set <base href="${CORE_PATH}"> so every relative URL in the page
 *      resolves against the real asset path, NOT the /u/(name)/ facade, and
 *   b) inject a one-liner that sets window.__SATELLITE_USER = "(name)" BEFORE
 *      the player's boot scripts run.
 * Non-HTML responses pass straight through.
 */
async function serveUserScoped(request, name, rest) {
  const url = new URL(request.url);
  // ALWAYS fetch the core player HTML from CORE_ORIGIN, regardless of the
  // request's incoming origin. This makes twysted.afest.io/radio/u* and
  // radio.afest.io/u/(name)/ route through the same fetch target (the GH
  // Pages path /v2-beta/), bypassing the Worker-doesn't-recurse-on-
  // subrequests limitation that would otherwise force the twysted branch
  // to land on a non-existent /u/twysted/ folder.
  const coreUrl = `${CORE_ORIGIN}${CORE_PATH.replace(/\/$/, '')}${rest}${url.search}`;
  const coreReq = new Request(coreUrl, {
    method: request.method,
    headers: request.headers,
    body: ['GET', 'HEAD'].includes(request.method) ? undefined : request.body,
    redirect: 'manual',
  });
  const coreResp = await fetch(coreReq);
  const contentType = coreResp.headers.get('content-type') || '';
  if (!contentType.toLowerCase().includes('text/html')) {
    return coreResp;
  }
  // Safe JSON-encoded user name — handles quotes, backslashes, anything.
  const scriptPayload = `window.__SATELLITE_USER = ${JSON.stringify(name)};`;
  // Absolute base URL so the same injected HTML behaves identically whether
  // it's served directly on radio.afest.io/u/(name)/ OR reverse-proxied under
  // twysted.afest.io/radio/u/. Subresource loads go directly to radio.afest.io
  // regardless of the address-bar origin.
  const baseHref = `${CORE_ORIGIN}${CORE_PATH}`;
  return new HTMLRewriter()
    .on('head', {
      element(el) {
        el.prepend(`<base href="${baseHref}">`, { html: true });
        el.prepend(`<script>${scriptPayload}</script>`, { html: true });
      },
    })
    .transform(coreResp);
}

/* ─────────────────────────────────────────────────────────────────────────
 * Verification checklist (run from any shell once the Worker is deployed):
 *
 *   curl -sI https://radio.afest.io/                   # 200  splash
 *   curl -sI https://radio.afest.io/v2-beta/           # 200  player HTML
 *   curl -sI https://radio.afest.io/u/twysted/         # 200  edge-rewrite
 *   curl -s  https://radio.afest.io/u/twysted/ | grep -o 'window.__SATELLITE_USER = "twysted"'
 *   curl -sI https://radio.afest.io/twysted            # 302  → /u/twysted/
 *   curl -sI https://radio.afest.io/twisted            # 302  → /u/twysted/ (alias)
 *   curl -sI https://radio.afest.io/wsi-rx/            # 302  → twysted.afest.io/radio
 *   curl -sI https://radio.afest.io/wsi-rx-stable/     # 200  v1.5 snapshot
 *   curl -sI https://radio.afest.io/fonts/WDXLLubrifontJPNVNM-Regular.woff2
 *                                                      # 200  reserved prefix
 *   curl -sI https://radio.afest.io/djsickmode         # 302  → /u/djsickmode/
 *   curl -sI https://radio.afest.io/u/ghost/           # 200  (player falls
 *                                                      # back to main view)
 *
 *   # Phase 1c proxy verification (twysted.afest.io → radio.afest.io):
 *   curl -sI https://twysted.afest.io/radio/u          # 200  proxy
 *   curl -sI https://twysted.afest.io/radio/u/         # 200  proxy
 *   curl -s  https://twysted.afest.io/radio/u/ | grep -o 'window.__SATELLITE_USER = "twysted"'
 *   curl -s  https://twysted.afest.io/radio/u/ | grep -o '<base href="[^"]*"'
 *                                                      # → <base href="https://radio.afest.io/v2-beta/"
 * ───────────────────────────────────────────────────────────────────────── */
