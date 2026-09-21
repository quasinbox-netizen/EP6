// The site is for people who were given the password, and nobody else.
//
// This runs on Cloudflare in front of every request, before any file is
// served - a page that checked the password in the browser would have
// already handed the page over. The password is the SITE_PASSWORD secret of
// the Pages project; it never appears in the repository.
//
// A correct password earns a cookie holding an HMAC of the password, not the
// password: changing SITE_PASSWORD logs everybody out, and a stolen cookie
// says nothing about what the password was.
//
// The cookie is SameSite=None; Partitioned, because the site is shown inside
// a frame on quasipi.tech, which makes it a third-party cookie. Partitioned
// (CHIPS) is the form browsers still accept there: the cookie is kept for
// this site *inside quasipi.tech* only. Where a browser refuses even that,
// the login page offers the site in a tab of its own.

const COOKIE = "btclab_auth";
const LOGIN_PATH = "/__login";
const MAX_AGE = 60 * 60 * 24 * 30;

async function token(password) {
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(password),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"],
  );
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode("btclab-v1"));
  return [...new Uint8Array(mac)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// Compared in constant time, so the response time does not leak how much of
// a guess was right.
function same(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

function cookieValue(request) {
  const header = request.headers.get("Cookie") || "";
  for (const part of header.split(";")) {
    const [name, ...rest] = part.trim().split("=");
    if (name === COOKIE) return rest.join("=");
  }
  return "";
}

// Only a path on this site, so the login cannot be used to bounce a visitor
// somewhere else.
function safeNext(value) {
  return typeof value === "string" && value.startsWith("/") && !value.startsWith("//")
    ? value : "/";
}

function loginPage(next, wrong) {
  const html = `<!doctype html>
<html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>BTC Cycle Lab</title>
<style>
  :root{--bg:#000308;--panel:#101219;--line:#252935;--ink:#f4f6f9;--dim:#8f99a8;--gold:#f7931a;--down:#ff5a5f}
  *{box-sizing:border-box}
  body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--ink);
    font:17px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI Variable Text","Segoe UI",Roboto,Arial,sans-serif;padding:16px}
  form{width:100%;max-width:360px;background:var(--panel);border-radius:16px;padding:28px 24px}
  h1{margin:0 0 6px;font-size:26px}
  p{margin:0 0 18px;color:var(--dim);font-size:15px}
  input{width:100%;padding:12px 14px;border-radius:10px;border:1px solid var(--line);background:#0a0d13;
    color:var(--ink);font-size:17px}
  button{margin-top:14px;width:100%;padding:12px;border:0;border-radius:999px;background:var(--gold);
    color:#0a0600;font-weight:700;font-size:16px;cursor:pointer}
  .wrong{color:var(--down);margin:10px 0 0}
  a{color:var(--gold);font-size:14px}
  .tab{margin:16px 0 0;text-align:center}
</style></head><body>
<form method="post" action="${LOGIN_PATH}">
  <h1>BTC Cycle Lab</h1>
  <p>Strona chroniona hasłem. &middot; This site is password protected.</p>
  <input type="hidden" name="next" value="${next.replace(/"/g, "&quot;")}">
  <input type="password" name="password" placeholder="Hasło / Password" autocomplete="current-password" autofocus required>
  ${wrong ? '<p class="wrong">Złe hasło. &middot; Wrong password.</p>' : ""}
  <button type="submit">Wejdź / Enter</button>
  <p class="tab"><a href="${next.replace(/"/g, "&quot;")}" target="_blank" rel="noopener">Otwórz w nowej karcie / Open in a new tab</a></p>
</form>
<script>
  // Tell the frame on quasipi.tech how tall this is, like every other page.
  try { parent.postMessage({ btclabHeight: document.documentElement.scrollHeight }, "*"); } catch (e) {}
</script>
</body></html>`;
  return new Response(html, {
    status: 401,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Robots-Tag": "noindex, nofollow",
    },
  });
}

export async function onRequest(context) {
  const { request, env, next } = context;
  const password = env.SITE_PASSWORD;
  if (!password) {
    // Fail closed: a missing secret must not mean an open site.
    return new Response("Site password is not configured.", { status: 503 });
  }
  const url = new URL(request.url);
  const expected = await token(password);

  if (url.pathname === LOGIN_PATH && request.method === "POST") {
    const form = await request.formData();
    const target = safeNext(form.get("next"));
    const given = String(form.get("password") || "");
    if (!same(await token(given), expected)) {
      // A pause on every wrong guess makes guessing slow without a database.
      await new Promise((resolve) => setTimeout(resolve, 900));
      return loginPage(target, true);
    }
    return new Response(null, {
      status: 303,
      headers: {
        Location: target,
        "Set-Cookie": `${COOKIE}=${expected}; Path=/; Max-Age=${MAX_AGE}; HttpOnly; Secure; SameSite=None; Partitioned`,
        "Cache-Control": "no-store",
      },
    });
  }

  if (!same(cookieValue(request), expected)) {
    return loginPage(safeNext(url.pathname + url.search), false);
  }

  const response = await next();
  const guarded = new Response(response.body, response);
  guarded.headers.set("X-Robots-Tag", "noindex, nofollow");
  guarded.headers.set("Cache-Control", "private, no-cache");
  return guarded;
}
