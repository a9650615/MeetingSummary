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


def is_on(store):
    """Feature enabled? The `remote_store` setting (UI toggle) OR the env flag.
    Default OFF so a normal build is dormant even with the folder present."""
    return enabled() or (store is not None
                         and store.get_setting("remote_store", "0") == "1")


def register(app, store):
    """Add the push route. Registered whenever the plugin is present, but the
    handler refuses unless is_on() — so a release can't push until toggled on.
    The detail-page button is gated on the same is_on() at render time."""
    from fastapi import HTTPException
    from plugins.remote_store import push as _push

    @app.post("/remote/push/{mid}")
    def remote_push(mid: int):
        if not is_on(store):
            raise HTTPException(404, "remote store disabled")
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        res = _push.build_and_push(store, mid, VM_URL)
        if not res["ok"]:
            raise HTTPException(502, f"push failed ({res['status']})")
        return res
