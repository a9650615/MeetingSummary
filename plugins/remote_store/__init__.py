"""Opt-in remote-store plugin. Gated by an env FLAG, default OFF — a normal
release ships this folder but the feature stays dormant (no push button, no
/remote route) unless REMOTE_STORE=1 is set. So the standard local build has
none of the server-side integration; only an explicitly-enabled machine does."""
import os

VM_URL = os.environ.get("REMOTE_STORE_URL", "http://10.102.0.7:5556")


def enabled():
    # default OFF: the feature is absent from a normal release unless the flag is
    # set (REMOTE_STORE=1, or a REMOTE_STORE_URL explicitly configured).
    return os.environ.get("REMOTE_STORE", "").strip() in ("1", "true", "on") \
        or bool(os.environ.get("REMOTE_STORE_URL_ENABLE"))


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
