"""Standalone remote-store server: read-only viewer + bundle ingest. Runs on the
VM, x86 Linux, no Apple/ASR deps. FireRed worker wired in a later task."""
import os
import tempfile
import zipfile

from fastapi import FastAPI, File, HTTPException, UploadFile

from store import Store
from viewer import bundle
from viewer.routes import mount_viewer

DB_PATH = os.environ.get("STORE_DB", "data/store.db")
DATA_DIR = os.environ.get("STORE_DATA", "data")


def build_server(db_path=None, data_dir=None):
    store = Store(db_path or DB_PATH)
    data = data_dir or DATA_DIR
    os.makedirs(data, exist_ok=True)
    app = FastAPI()
    app.state.store = store
    app.state.data_dir = data
    app.state.on_ingest = None  # Task 9 sets this to the FireRed worker's enqueue
    mount_viewer(app, store, data)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/ingest-bundle")
    async def ingest_bundle(bundle_file: UploadFile = File(..., alias="bundle")):
        raw = await bundle_file.read()
        with tempfile.TemporaryDirectory() as td:
            zp = os.path.join(td, "in.zip")
            with open(zp, "wb") as f:
                f.write(raw)
            try:
                bd, tracks = bundle.read_bundle_zip(zp, os.path.join(td, "x"))
                mid = bundle.ingest_bundle(store, data, bd, tracks)
            except (zipfile.BadZipFile, KeyError, ValueError) as e:
                raise HTTPException(400, f"bad bundle: {e}")
        if app.state.on_ingest:
            app.state.on_ingest(mid)
        return {"mid": mid}

    return app


app = build_server()
