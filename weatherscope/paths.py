"""Central place for every file path used by the project.

All paths are built from the location of THIS file, so the project works no
matter which folder you start Python or Streamlit from, and no absolute paths
(like C:\\Users\\...) are ever hard-coded.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "weather_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

METRICS_PATH = REPORT_DIR / "metrics.json"
PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"
