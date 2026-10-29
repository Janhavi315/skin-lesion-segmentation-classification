# Skin-Lesion Segmentation & Classification (ML + DL)

A beginner-friendly Streamlit college mini-project for approximate skin-lesion segmentation, image-quality heuristics, image-based area estimation, HAM10000 category classification using trained SVM / Random Forest / small CNN models, model evaluation, and downloadable PDF reports.

> **Medical disclaimer:** This is an educational prototype, not a medical device or diagnostic tool. It cannot determine whether a lesion is cancerous or harmless. Model confidence is not the probability a patient has a disease. Seek professional medical evaluation for concerning or changing lesions.

## Included features

1. **Approximate segmentation:** OpenCV GrabCut mask and contour overlay. This is classical image processing, not a clinically validated segmentation model.
2. **Classification:** SVM and Random Forest use compact color/texture features; a small CNN learns from resized images. The app never invents predictions: saved trained artifacts are required.
3. **Model comparison:** Accuracy, macro precision/recall/F1 and confusion matrices from actual held-out predictions.
4. **Image-quality checks:** documented thresholds for darkness, blur and resolution. These do not guarantee medical suitability.
5. **Area estimate:** segmented pixels and percentage of image area, without claiming physical units.
6. **Educational indicator:** a transparent category-based flag, not a clinical risk score.
7. **Dynamic PDF report:** current image, segmentation outputs, quality checks, area and available predictions/evaluation metrics.

## Requirements / laptop notes

- Windows 64-bit; CPU-only operation is supported; no NVIDIA GPU is required.
- Python **3.10 or 3.11 (64-bit)** is recommended for the pinned TensorFlow 2.15.1 dependency. Python 3.14 is not compatible with this TensorFlow pin. Install Python 3.11 if needed.
- 8 GB RAM: start with the default 2,500-image subset, 64×64 CNN input, batch size 16 and 5 epochs. Training time depends on the laptop and dataset. If memory is tight, use `--limit 800`.
- The dataset is not included in this ZIP. Model files are not included because no local dataset was available for training/evaluation.

## 1. Install Python and open the project

Install 64-bit Python 3.11 from the official Python downloads page. During setup, enable **Add Python to PATH**. Open this folder in VS Code.

Open **Terminal → New Terminal** (PowerShell) and check:

```powershell
py -0p
py -3.11 --version
```

If `py -3.11` is not recognized, install Python 3.11 and reopen the terminal.

## 2. Create and activate a virtual environment

Run these commands from the folder containing `app.py`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use this only for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Wait for TensorFlow and the other packages to finish installing. If installation fails, confirm that the terminal is using Python 3.11 64-bit.

## 3. Put HAM10000 in the expected folders

Download HAM10000 from its official/reputable dataset source separately and follow its terms of use. Do **not** put the full dataset into this project ZIP or upload private patient data.

Default expected layout:

```text
skin_lesion_project/
  data/
    HAM10000/
      HAM10000_metadata.csv
      HAM10000_images/
        ISIC_0024306.jpg
        ...
```

The exact image names are read from the `image_id` column in the metadata CSV; `dx` is the diagnostic label. If your images are split across two folders, either combine the image files into `HAM10000_images` or configure both folders.

### Configure a custom path (PowerShell)

The defaults are in `config.py`. You can change `DATA_DIR`, `METADATA_CSV`, and `IMAGE_DIRS` there. Alternatively set environment variables before running commands:

```powershell
$env:HAM10000_DATA_DIR = "D:\datasets\HAM10000"
$env:HAM10000_METADATA_CSV = "D:\datasets\HAM10000\HAM10000_metadata.csv"
$env:HAM10000_IMAGE_DIRS = "D:\datasets\HAM10000\HAM10000_images;D:\datasets\HAM10000\HAM10000_images_part2"
```

On Windows, `os.pathsep` is a semicolon, so separate multiple image directories with `;`. Keep the metadata CSV and image folders consistent.

## 4. Verify the dataset before training

```powershell
python scripts/check_dataset.py
```

The report checks whether the metadata exists, required columns are present, image folders exist, metadata IDs match image files, category counts, and optionally whether images can be opened. If the metadata CSV is missing, put it at the configured path. If images are missing, ensure each `image_id` has a matching `.jpg`, `.jpeg`, or `.png` file in one configured image folder.

## 5. Train the models (separate from website startup)

Train SVM and Random Forest with the default subset:

```powershell
python scripts/train_models.py
```

Train on a smaller subset if needed:

```powershell
python scripts/train_models.py --limit 800
```

Train SVM, Random Forest **and** CNN:

```powershell
python scripts/train_models.py --limit 2500 --cnn --epochs 5
```

CNN training runs on CPU and may take a while. The training script saves models to `models/` and actual evaluation results to `outputs/evaluation.json`. Do not expect a particular accuracy before running it on your dataset. If training fails because a class has too few examples for a stratified split, use a larger subset or inspect your metadata distribution.

## 6. Start the website

```powershell
streamlit run app.py
```

Streamlit normally opens a browser tab automatically. The app is local on your computer. It starts even if models are not trained yet and shows instructions instead of generating fake predictions.

## 7. Use the website

- **Analyze Image:** upload a JPG/JPEG/PNG, inspect original / mask / contour overlay, read quality warnings and area estimate, choose an available trained model, and prepare/download a PDF report.
- **Model Comparison:** view metrics and confusion matrices recorded by the training process. Results are only displayed after training.
- **Dataset & Model Information:** run dataset checks, inspect metadata category counts, and see saved model artifacts.

## Troubleshooting

- **`py -3.11` not found:** install Python 3.11 64-bit, then reopen VS Code.
- **TensorFlow install error:** activate `.venv` and confirm `python --version` is 3.11; the pinned TensorFlow version does not support Python 3.14.
- **Metadata not found:** place `HAM10000_metadata.csv` at the configured path or update `config.py` / environment variables.
- **Images missing:** the image filenames should match `image_id` exactly, with a supported image extension. Configure every image folder.
- **App says no trained model:** run the training command after dataset verification.
- **Slow training / high memory use:** use `--limit 800`; close memory-heavy applications. The app does not retrain on startup.
- **OpenCV GrabCut error / empty mask:** try a well-lit, larger image with the lesion reasonably centered. Segmentation is approximate and may fail.
- **PDF button:** click **Prepare PDF report** first; then click **Download PDF report**.
- **TensorFlow / CNN not available:** train only SVM and Random Forest by omitting `--cnn`.

## Project structure

```text
skin_lesion_project/
  app.py
  config.py
  requirements.txt
  README.md
  src/
    dataset.py
    preprocessing.py
    segmentation.py
    image_quality.py
    features.py
    train_ml.py
    train_cnn.py
    prediction.py
    evaluation.py
    report.py
  scripts/
    check_dataset.py
    train_models.py
  data/HAM10000/       # dataset goes here; not included
  models/              # trained models are created here
  outputs/             # metrics and report artifacts are created here
```

## Limitations / what has not been tested here

- No HAM10000 dataset was supplied with the project request, so model training, accuracy numbers and a real sample analysis report could not be produced honestly.
- The source files are checked for Python syntax in this package build, but you should run dataset verification and training on your machine.
- This project is for learning and demonstration only. It is not clinically validated and must not be used for diagnosis or treatment decisions.
# skin-lesion-segmentation-classification
Skin lesion segmentation and classification using machine learning and deep learning with a Streamlit interface.
