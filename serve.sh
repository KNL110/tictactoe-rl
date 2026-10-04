#!/usr/bin/env bash
# Starts the Tic-Tac-Toe web app plus a Cloudflare quick tunnel, so your phone can
# reach it over the public internet (cellular data, different/weak wifi) instead of
# needing to be on the exact same local network as this machine.
#
# The printed https://*.trycloudflare.com URL is randomly generated and only live
# while this script runs — anyone with the link could open it in that window, but
# it's not guessable and dies the moment you Ctrl-C this script.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

# TTT_PERSIST=1: games you play here keep training the saved models (never set on the
# public Hugging Face deployment, see webapp/game.py).
TTT_PERSIST=1 python -m webapp > /tmp/tictactoe_webapp.log 2>&1 &
WEBAPP_PID=$!

cleanup() {
    echo -e "\nStopping..."
    kill "$WEBAPP_PID" "$TUNNEL_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

sleep 1
~/.local/bin/cloudflared tunnel --url http://localhost:5000 > /tmp/tictactoe_tunnel.log 2>&1 &
TUNNEL_PID=$!

echo "Starting tunnel..."
URL=""
for i in $(seq 1 30); do
    URL=$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' /tmp/tictactoe_tunnel.log | head -1)
    if [ -n "$URL" ]; then
        break
    fi
    sleep 1
done

if [ -z "$URL" ]; then
    echo "Tunnel didn't come up in time — check /tmp/tictactoe_tunnel.log"
else
    echo ""
    echo "Open this on your phone (works on ANY network, not just this wifi):"
    echo "  $URL"
    echo ""
    echo "Press Ctrl-C here to stop the server and tunnel."
fi

wait "$TUNNEL_PID"
