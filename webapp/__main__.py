"""Local dev server: `python -m webapp`, then open the printed URL on your phone
(same wifi as this machine). Set PORT to change the port (default 5000)."""
import os
import socket

from webapp.app import app


def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nOpen this on your phone (same wifi as this machine): http://{local_ip()}:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
