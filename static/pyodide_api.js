// GitHub Pages build only (inserted into index.html by build_static.py).
// Runs the web app's own Python code (webapp/api.py, webapp/game.py, tictactoe/)
// in the browser with Pyodide, so the page needs no server. index.html's api()
// sends its calls to window.localApi instead of fetch() when this is present.
(() => {
  const statusEl = document.getElementById('status');
  const newGameBtn = document.getElementById('newGameBtn');
  newGameBtn.disabled = true;
  statusEl.textContent = 'Loading the agents… (first visit downloads ~10 MB)';

  async function writeFile(pyodide, path) {
    const res = await fetch('py/' + path);
    if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
    const parts = ('app/' + path).split('/');
    for (let i = 1; i < parts.length; i++) {
      const dir = parts.slice(0, i).join('/');
      if (!pyodide.FS.analyzePath(dir).exists) pyodide.FS.mkdir(dir);
    }
    pyodide.FS.writeFile('app/' + path, new Uint8Array(await res.arrayBuffer()));
  }

  window.localApiReady = (async () => {
    const [pyodide, files] = await Promise.all([
      loadPyodide({packages: ['numpy']}),
      fetch('py/files.json').then(r => r.json()),
    ]);
    await Promise.all(files.map(path => writeFile(pyodide, path)));
    pyodide.runPython(`
import json, sys
sys.path.insert(0, "app")
from webapp import api
from webapp.game import GameSession

_session = GameSession()  # this browser is the only visitor

def call(path, body_json):
    status, body = api.handle(path, json.loads(body_json), _session)
    return json.dumps(body)
`);
    const call = pyodide.globals.get('call');
    window.localApi = async (path, body) => JSON.parse(call(path, JSON.stringify(body)));
  })();

  window.localApiReady.then(
    () => {
      newGameBtn.disabled = false;
      statusEl.textContent = 'Press New Game to start';
    },
    err => {
      console.error(err);
      statusEl.textContent = 'Failed to load the agents. Check your connection and reload.';
    },
  );
})();
