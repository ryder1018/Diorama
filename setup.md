# Running the phone → gateway → backend loop

The Unity app on a phone talks to a small gateway on the Windows PC, which forwards each photo to the level API in WSL:

```
Unity app ──► win-server/server.py (Windows, :5000) ──► wsl-cv/scripts/infer.py (WSL, :9000)
```

## 1. Level API (WSL)

In the Python env from the root [`README.md`](README.md#installation), with `MODEL_API_KEY` set (or in the repo-root `.env`):

```bash
cd wsl-cv
set -a; . ../.env; set +a        # if the key is in .env
uvicorn scripts.infer:app --host 0.0.0.0 --port 9000
```

From Windows, `curl.exe http://127.0.0.1:9000/health` should answer `{"status":"ok",...}`. If it does not, WSL's localhost forwarding is off; use the WSL address (`hostname -I` in WSL) as `SKATEP_WSL_URL` below.

## 2. Gateway (Windows)

The gateway needs Python with `flask` and `requests`:

```powershell
cd win-server
pip install flask requests
python server.py
```

It listens on port 5000. Settings are environment variables:

| Variable | Default | |
|---|---|---|
| `SKATEP_API_KEY` | `0000` | must match `apiKey` in `MobileCameraCapture.cs` |
| `SKATEP_WSL_URL` | `http://127.0.0.1:9000/infer` | the level API |
| `SKATEP_WSL_TIMEOUT` | `300` | seconds to wait for one level |

For a phone, allow inbound TCP 5000 in Windows Firewall and enter the PC's Wi-Fi IPv4 address (`ipconfig`) in the app's server IP field; the app remembers it. A phone cannot reach a gateway running inside WSL (NAT), but for Unity Editor tests on the same PC the gateway can also run in WSL (`cd win-server && python server.py`).

## 3. Try the upload flow by hand

This is what the app does: upload, then poll until the level is ready (about 1.5 minutes).

```powershell
curl.exe -X POST http://127.0.0.1:5000/upload -H "X-API-KEY: 0000" -F "photo=@C:\path\to\photo.jpg;type=image/jpeg" -F "style=comic"
# {"status":"accepted"}

curl.exe http://127.0.0.1:5000/download -H "X-API-KEY: 0000" -o result.json
# {"status":"processing"} until it becomes {"status":"success","images":{...},"metadata":{...}}
```

`style` is optional: `original`, `comic`, `cyberpunk`, `snow`, `pixel` or `watercolor` (the app's map style picker sends it). A failed level answers `{"status":"error","message":"..."}`. The response format is documented in [`wsl-cv/README.md`](wsl-cv/README.md#api).

## 4. Unity Editor smoke test

With the API and the gateway running and `Parallax.unity` open, **Tools > SkateP > Pipeline Smoke Test...** picks a photo, enters Play mode, uploads it through `MobileCameraCapture` with the style chosen in the picker, waits for the result panel, presses Start Game, checks that the generated level is the one in play, and writes screenshots and `report.txt` to `Temp/PipelineSmokeTest/` in the Unity project (`LocalLevelTester`'s baked levels are skipped for that run).

It can also be started without touching the Editor by writing `Temp/pipeline_smoke_test.json` in the Unity project:

```json
{"photo": "C:/Users/Public/skatep_test/park.jpg", "style": "cyberpunk", "server": "127.0.0.1"}
```

An empty `Temp/pipeline_smoke_refresh` file makes the Editor re-import scripts edited outside Unity.
