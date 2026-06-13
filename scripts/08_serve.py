"""Launch the FastAPI reasoning service.

Variant/overrides come from env (AXIOM_VARIANT, AXIOM_OVERRIDES); see serve/app.py.
"""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("axiom.serve.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
