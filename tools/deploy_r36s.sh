#!/bin/bash
# Deploy the Midnight Club port to the R36S over SSH.
#
# Usage: deploy_r36s.sh [host]
#
# Pushes build-r36s/src/mc3/mc3 plus the launch script and port.json into
# /roms2/ports/midnight-club-lar/ on the device, then runs it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${1:-${R36S_HOST:-192.168.2.121}}"
USER="${R36S_USER:-ark}"
PROJ="${MC3_PROJ:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PORT="${R36S_PORT_DIR:-/roms2/ports/midnight-club-lar}"
SSH=(ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes "$USER@$HOST")
SCP=(scp -o StrictHostKeyChecking=accept-new -o BatchMode=yes)

BIN="$PROJ/build-r36s/src/mc3/mc3"
[ -f "$BIN" ] || { echo "ERROR: build not found: $BIN"; exit 1; }

echo "=== Deploying to $HOST:$PORT ==="
"${SSH[@]}" "mkdir -p '$PORT'"

"${SCP[@]}" "$BIN" "$USER@$HOST:$PORT/mc3"
"${SCP[@]}" "$PROJ/ports/midnight-club-lar/Midnight Club - L.A. Remix.sh" "$USER@$HOST:$PORT/"
"${SCP[@]}" "$PROJ/ports/midnight-club-lar/port.json" "$USER@$HOST:$PORT/"

"${SSH[@]}" "chmod +x '$PORT/mc3' '$PORT/Midnight Club - L.A. Remix.sh'"

echo "=== Verifying ==="
"${SSH[@]}" "ls -la '$PORT'; echo; file '$PORT/mc3' 2>/dev/null || true; echo; readelf --version-info '$PORT/mc3' 2>/dev/null | grep -oE 'GLIBC_2\.[0-9]+' | sort -uV || true"

echo
echo "=== Run (30s) ==="
"${SSH[@]}" "cd '$PORT' && timeout 30 './Midnight Club - L.A. Remix.sh' 2>&1 | tail -50; echo \"exit=\$?\""
