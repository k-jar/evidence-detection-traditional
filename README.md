# Evidence Detection using XGBoost and Sentence Embeddings

## 1. Overview

This project implements a binary classifier to determine if a given piece of `Evidence` is relevant to a `Claim`. It uses an XGBoost model trained on a combination of traditional NLP features and sentence embeddings.

## 2. Implementation Details

*   **Task:** Pairwise sequence classification (Evidence Detection).
*   **Model:** XGBoost (`xgboost.XGBClassifier`), tuned with Optuna.
*   **Features:**
    *   **Traditional:** Jaccard similarity, VADER sentiment, text lengths/ratios, word counts/ratios, word overlap.
    *   **Embeddings:** `SentenceTransformer` (`all-MiniLM-L6-v2`) embeddings for claim and evidence, combined via difference and element-wise product.
*   **Libraries:** `xgboost`, `sentence-transformers`, `optuna`, `nltk`, `scikit-learn`, `pandas`, `numpy`.

## 3. Setup

1.  **Environment:** Python >3.10.
2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy optuna scikit-learn nltk matplotlib seaborn tqdm xgboost sentence-transformers kaleido
    ```

## 4. Usage

The Jupyter notebook contains configuration options at the top.

*   **To Train a New Model:**
    1.  Set `USE_PRETRAINED_MODEL = False`.
    2.  Configure `DATA_PATH`, `DEV_DATA_PATH`.
    3.  Ensure the last cell calls `main()`.
    4.  Run the notebook. Saves model to `.pkl` file.

*   **To Generate Predictions (Inference Mode):**
    1.  Set `USE_PRETRAINED_MODEL = True`.
    2.  Set `MODEL_PATH` to your saved `.pkl` model.
    3.  Set `TEST_DATA_PATH` to the **unlabeled** input CSV (no header, 2 columns: Claim, Evidence).
    4.  Set `PREDICTIONS_OUTPUT_PATH` for the output CSV.
    5.  Ensure the last cell calls `run_inference()`.
    6.  Run the notebook. Generates predictions file.

*   **To Evaluate a Model:**
    1.  Set `USE_PRETRAINED_MODEL = True`.
    2.  Set `MODEL_PATH` to your saved `.pkl` model.
    3.  Set `TEST_DATA_PATH` to a **labeled** CSV (e.g., dev set).
    4.  Ensure the last cell calls `main()`.
    5.  Run the notebook. Prints evaluation metrics.

## 5. Data Format

*   **Input (Training/Dev):** CSV with headers `Claim`, `Evidence`, `label`.
*   **Input (Inference):** CSV without headers, Col 1: `Claim`, Col 2: `Evidence`.
*   **Output (Predictions):** CSV with header `predictions`, containing predicted labels (0 or 1).

## 6. Model Storage

*   **Link to Model:** [Google Drive](https://drive.google.com/drive/folders/1roXEaI_bne7Vlwe_xuRYL0mPA8lUZaL6?usp=drive_link)

## 7. Attribution

*   [XGBoost](https://github.com/dmlc/xgboost) (Classifier)
*   [Sentence Transformers](https://www.sbert.net/) (Embeddings: `all-MiniLM-L6-v2`)
*   [Optuna](https://optuna.org/) (Hyperparameter Tuning)
*   [NLTK](https://www.nltk.org/) (Preprocessing, VADER)
*   [Scikit-learn](https://scikit-learn.org/) (Metrics, CV)
*   [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Tqdm](https://tqdm.github.io/), [Kaleido](https://github.com/plotly/Kaleido)
