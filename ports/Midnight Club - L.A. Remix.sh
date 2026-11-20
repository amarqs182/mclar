#!/bin/sh
# PortMaster indexes launchers from the ports/ directory. Keep the port's
# assets and binary in their own directory and delegate to its launcher.
PORTS_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$PORTS_DIR/midnight-club-lar/Midnight Club - L.A. Remix.sh" "$@"
