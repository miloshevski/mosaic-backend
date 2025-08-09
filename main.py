import os
import cv2
import zipfile
import tempfile
import numpy as np
from glob import glob
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

app = FastAPI()

# ✅ CORS: дозволи локален дев + Vercel
origins = [
    "http://localhost:3000",
    "https://your-frontend.vercel.app",  # смени со твојот точен домен
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # ["*"] само за тест
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Utility =========
def average_lab(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).mean(axis=0)

def load_tiles(tiles_dir: str, tile_size: int):
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        paths.extend(glob(os.path.join(tiles_dir, ext)))
    if not paths:
        raise FileNotFoundError("No tile images found in tiles_dir")

    tiles_small, tiles_lab = [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        min_side = min(h, w)
        y0 = (h - min_side) // 2
        x0 = (w - min_side) // 2
        img_cropped = img[y0:y0+min_side, x0:x0+min_side]
        img_resized = cv2.resize(img_cropped, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        tiles_small.append(img_resized)
        tiles_lab.append(average_lab(img_resized))

    if not tiles_small:
        raise RuntimeError("Failed to load any valid tile images.")

    return np.array(tiles_small), np.array(tiles_lab, dtype=np.float32)

def build_mosaic_bytes(
    target_bytes: bytes,
    tiles_dir: str,
    tile_size: int = 16,
    blend: float = 0.12,
    no_immediate_repeat: bool = True,
    max_width: int = 800,
) -> bytes:
    file_bytes = np.frombuffer(target_bytes, np.uint8)
    target_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if target_original is None:
        raise FileNotFoundError("Can't read target image")

    target = target_original.copy()
    if target.shape[1] > max_width:
        scale = max_width / target.shape[1]
        new_size = (max_width, int(target.shape[0] * scale))
        target = cv2.resize(target, new_size, interpolation=cv2.INTER_AREA)

    h, w = target.shape[:2]
    h_new = (h // tile_size) * tile_size
    w_new = (w // tile_size) * tile_size
    target = cv2.resize(target, (w_new, h_new), interpolation=cv2.INTER_AREA)

    tiles_small, tiles_lab = load_tiles(tiles_dir, tile_size)

    mosaic = np.zeros_like(target)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    rows = h_new // tile_size
    cols = w_new // tile_size
    last_used_idx = -1

    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * tile_size, (r + 1) * tile_size
            x0, x1 = c * tile_size, (c + 1) * tile_size

            patch_lab = target_lab[y0:y1, x0:x1]
            mean_lab = patch_lab.reshape(-1, 3).mean(axis=0)

            dists = np.linalg.norm(tiles_lab - mean_lab, axis=1)
            if no_immediate_repeat and last_used_idx >= 0:
                dists[last_used_idx] += 5.0

            idx = int(np.argmin(dists))
            mosaic[y0:y1, x0:x1] = tiles_small[idx]
            last_used_idx = idx

    if blend > 0:
        mosaic = cv2.addWeighted(mosaic, 1 - blend, target, blend, 0)

    ok, buf = cv2.imencode('.jpg', mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode mosaic to JPEG")
    return buf.tobytes()

# ========= Extra routes =========
@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/api/health", "/api/mosaic", "/docs"]}

@app.get("/api/health")
def health():
    return {"ok": True}

# ========= API endpoint =========
@app.post("/api/mosaic")
async def make_mosaic(
    target: UploadFile = File(...),
    tiles_zip: UploadFile | None = File(None),
    tiles_files: list[UploadFile] | None = None,
    tile_size: int = Form(16),
    blend: float = Form(0.12),
    no_immediate_repeat: bool = Form(True),
    max_width: int = Form(800),
):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiles_dir = os.path.join(tmpdir, "tiles")
            os.makedirs(tiles_dir, exist_ok=True)

            if tiles_zip is not None:
                zbytes = await tiles_zip.read()
                zpath = os.path.join(tmpdir, "tiles.zip")
                with open(zpath, "wb") as f:
                    f.write(zbytes)
                with zipfile.ZipFile(zpath, 'r') as zf:
                    zf.extractall(tiles_dir)

            if tiles_files:
                for up in tiles_files:
                    b = await up.read()
                    fname = os.path.basename(up.filename or "tile.jpg")
                    with open(os.path.join(tiles_dir, fname), "wb") as f:
                        f.write(b)

            mosaic_jpg = build_mosaic_bytes(
                target_bytes=await target.read(),
                tiles_dir=tiles_dir,
                tile_size=tile_size,
                blend=blend,
                no_immediate_repeat=no_immediate_repeat,
                max_width=max_width,
            )
            return Response(content=mosaic_jpg, media_type="image/jpeg")

    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Server error: {e}"}, status_code=500)
