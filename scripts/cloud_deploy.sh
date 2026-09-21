#!/usr/bin/env bash
# Put docs/ on Cloudflare Pages, behind the password in functions/_middleware.js.
#
# Deploys only when the published pages changed - the committed docs/ tree,
# which moves when the daily refresh rebuilds the site or the agent decides its
# feed is worth publishing - or when forced. Every hourly tick deploying would
# be seven hundred deployments a month for a journal line nobody reads twice.
#
# Needs CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID and SITE_PASSWORD. Without
# them it says so and stops, rather than failing every run.
set -euo pipefail

PROJECT="${CF_PAGES_PROJECT:-quasipi-btc-lab}"
MARKER=data/processed/cloudflare_deployed.txt
WRANGLER=(npx --yes wrangler@4)

for name in CLOUDFLARE_API_TOKEN CLOUDFLARE_ACCOUNT_ID SITE_PASSWORD; do
  if [ -z "${!name:-}" ]; then
    echo "::warning::$name is not set; the site was not deployed."
    exit 0
  fi
done

tree="$(git rev-parse HEAD:docs)"
secret="$(printf '%s' "$SITE_PASSWORD" | sha256sum | cut -c1-16)"
wanted="$tree $secret"
if [ "${LAB_FORCE_DEPLOY:-false}" != "true" ] && [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$wanted" ]; then
  echo "published pages unchanged since the last deploy; nothing to do"
  exit 0
fi

if ! "${WRANGLER[@]}" pages project list 2>/dev/null | grep -q "$PROJECT"; then
  "${WRANGLER[@]}" pages project create "$PROJECT" --production-branch main
fi

# The password lives in Cloudflare, not in the files. Set again only when it
# changed; a new secret takes effect with the deployment that follows.
if [ ! -f "$MARKER" ] || [ "$(cut -d' ' -f2 "$MARKER")" != "$secret" ]; then
  printf '%s' "$SITE_PASSWORD" | "${WRANGLER[@]}" pages secret put SITE_PASSWORD --project-name "$PROJECT"
fi

# From the repository root, so wrangler picks up functions/ next to docs/.
"${WRANGLER[@]}" pages deploy docs --project-name "$PROJECT" --branch main \
  --commit-hash "$(git rev-parse HEAD)" --commit-dirty=true
echo "$wanted" > "$MARKER"
