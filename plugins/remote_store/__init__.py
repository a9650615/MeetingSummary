"""Opt-in remote-store plugin. Present == feature on. A minimal try-import seam
in app.py calls register(); the base build ships without this folder."""
import os

VM_URL = os.environ.get("REMOTE_STORE_URL", "http://10.102.0.7:5556")


def enabled():
    return True  # presence of this package is the switch


def register(app, store):
    """Add the push route. The detail-page button is injected by the app.py seam
    guarded on this module importing."""
    from fastapi import HTTPException
    from plugins.remote_store import push as _push

    @app.post("/remote/push/{mid}")
    def remote_push(mid: int):
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        res = _push.build_and_push(store, mid, VM_URL)
        if not res["ok"]:
            raise HTTPException(502, f"push failed ({res['status']})")
        return res
