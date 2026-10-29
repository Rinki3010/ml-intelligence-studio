from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
DATA_DIR = PROJECT_DIR / "data"
DATABASE_PATH = BACKEND_DIR / "studio.sqlite3"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
