"""LITE backend for the sumo web UI.

A FastAPI app with NO authentication and NO database — it reads and writes
the local filesystem only (committed ``checkpoints/*.pt`` and a small
``data/`` cache). See :mod:`webapp.backend.app` for the ASGI app.
"""
