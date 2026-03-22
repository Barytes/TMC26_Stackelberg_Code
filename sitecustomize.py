from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

for path in (ROOT, SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

cache_root = Path(os.environ.get("TMC26_CACHE_DIR", "/tmp/tmc26_cache"))
mpl_cache = cache_root / "matplotlib"
xdg_cache = cache_root / "xdg"
mpl_cache.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
