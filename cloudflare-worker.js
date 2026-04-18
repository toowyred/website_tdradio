/**
 * assets.afest.io referrer-gating worker
 * ----------------------------------------------------------
 * Route:   assets.afest.io/twysted/radio/*
 * Bucket:  afest-assets  (R2 binding name: ASSETS)
 *
 * Blocks hotlinks from non-afest origins. Served files still go
 * directly from R2 — this worker just validates the Referer header
 * before allowing the asset through.
 *
 * Allowed referers:
 *   - https://twysted.afest.io/* (the player lives here)
 *   - https://radio.afest.io/*   (the splash page, in case it ever fetches)
 *   - https://afest.io/*         (any afest.io path, just in case)
 *   - Empty referer is ALLOWED for signal.dat ONLY (so direct curl during
 *     debugging works for the metadata file but not the audio).
 *
 * Deploy:
 *   1. Cloudflare dashboard → Workers & Pages → Create Worker
 *   2. Paste this file, save + deploy
 *   3. Bindings → R2 bucket: name=ASSETS, bucket=afest-assets
 *   4. Trigger: Add route  assets.afest.io/twysted/radio/*  → this worker
 *   5. Purge cache for that path in case of stale entries
 *
 * Notes:
 *   - wrangler.toml equivalent bindings included below (commented)
 *   - Range requests pass through so large WAVs stream correctly
 */

const ALLOWED_HOSTS = new Set([
  'twysted.afest.io',
  'radio.afest.io',
  'afest.io',
  'www.afest.io',
  // add localhost for dev if desired (DEV ONLY):
  // 'localhost:8080',
]);

const META_FILES = new Set([
  'signal.dat',   // allow empty-referer fetches of metadata for debugging
]);

function isAllowedReferer(refererHeader) {
  if (!refererHeader) return false;
  try {
    const u = new URL(refererHeader);
    return ALLOWED_HOSTS.has(u.host);
  } catch (_) { return false; }
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    // only handle the gated path
    const prefix = '/twysted/radio/';
    if (!url.pathname.startsWith(prefix)) {
      return new Response('Not found', { status: 404 });
    }

    const key = url.pathname.slice(1); // strip leading /
    const filename = key.split('/').pop();

    // referer check
    const referer = request.headers.get('Referer') || '';
    const allowed = isAllowedReferer(referer) || META_FILES.has(filename);

    if (!allowed) {
      return new Response('Forbidden: hotlink blocked.', {
        status: 403,
        headers: { 'Content-Type': 'text/plain; charset=utf-8' },
      });
    }

    // CORS for browser fetch()
    const origin = request.headers.get('Origin') || '';
    const corsOrigin = [...ALLOWED_HOSTS].find(h => origin.endsWith(h))
      ? origin
      : 'https://twysted.afest.io';

    // OPTIONS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 204,
        headers: {
          'Access-Control-Allow-Origin': corsOrigin,
          'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
          'Access-Control-Allow-Headers': 'Range',
          'Access-Control-Max-Age': '86400',
        },
      });
    }

    // fetch from R2
    const range = request.headers.get('range');
    const r2opts = {};
    if (range) {
      // parse bytes=N-M
      const m = /^bytes=(\d+)-(\d*)$/.exec(range);
      if (m) {
        const offset = parseInt(m[1], 10);
        const end = m[2] ? parseInt(m[2], 10) : undefined;
        r2opts.range = end !== undefined
          ? { offset, length: end - offset + 1 }
          : { offset };
      }
    }

    const object = await env.ASSETS.get(key, r2opts);
    if (!object) {
      return new Response('Not found', { status: 404 });
    }

    const headers = new Headers();
    object.writeHttpMetadata(headers);
    headers.set('etag', object.httpEtag);
    headers.set('Access-Control-Allow-Origin', corsOrigin);
    headers.set('Access-Control-Expose-Headers', 'Content-Range, Content-Length, Accept-Ranges');
    headers.set('Accept-Ranges', 'bytes');
    headers.set('Cache-Control', 'public, max-age=3600');
    // discourage caching by download managers that respect no-store
    if (/\.(wav|mp3|flac|ogg|m4a)$/i.test(filename)) {
      headers.set('Content-Disposition', 'inline');
    }

    if (range && object.range) {
      const start = object.range.offset || 0;
      const length = object.range.length || object.size - start;
      headers.set('Content-Range', `bytes ${start}-${start + length - 1}/${object.size}`);
      headers.set('Content-Length', String(length));
      return new Response(object.body, { status: 206, headers });
    }

    return new Response(object.body, { status: 200, headers });
  },
};

/* ──────────────────────────────────────────────────────────
   wrangler.toml equivalent:

   name = "afest-assets-gate"
   main = "cloudflare-worker.js"
   compatibility_date = "2025-01-01"

   [[r2_buckets]]
   binding = "ASSETS"
   bucket_name = "afest-assets"

   [[routes]]
   pattern = "assets.afest.io/twysted/radio/*"
   zone_name = "afest.io"
   ────────────────────────────────────────────────────────── */
