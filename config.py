"""Central settings for the skin-lesion educational mini-project."""
from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("HAM10000_DATA_DIR", PROJECT_DIR / "data" / "HAM10000"))
METADATA_CSV = Path(os.getenv("HAM10000_METADATA_CSV", DATA_DIR / "HAM10000_metadata.csv"))
# Configure one or more folders; add a second folder here if your images are split.
IMAGE_DIRS = [
    Path(p.strip()) for p in os.getenv(
        "HAM10000_IMAGE_DIRS",
        str(DATA_DIR / "HAM10000_images")
    ).split(os.pathsep) if p.strip()
]
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
IMAGE_SIZE = int(os.getenv("SKIN_IMAGE_SIZE", "64"))
DEFAULT_TRAIN_LIMIT = int(os.getenv("SKIN_TRAIN_LIMIT", "2500"))
TEST_SIZE = 0.20
RANDOM_SEED = 42
BATCH_SIZE = 16
CNN_EPOCHS = 5
CLASS_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
REQUIRED_COLUMNS = {"image_id", "dx"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
