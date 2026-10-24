# Mortgage-Prepayment-Prediction-
# Mortgage-Backed Securities Prepayment Prediction Using Machine Learning

Predicting whether a mortgage loan will be **prepaid**, using loan origination characteristics from the **Freddie Mac 2024 sample dataset**.

**Authors**
- Sanjeev Murthy B (PES1UG24CS612)
- Vishwas R Moodalappa (PES1UG24CS644)

---

## Overview

Mortgage-backed securities (MBS) depend on the cash flows of underlying home loans. When borrowers repay a loan earlier than scheduled, those cash flows change, which creates prepayment risk for investors. This project builds a binary classifier that predicts whether a loan is prepaid (`1`) or not (`0`) from information available when the loan is originated.

Four models are compared under identical preprocessing and a leakage-safe evaluation protocol:

- Logistic Regression
- RBF Support Vector Machine (SVM)
- Random Forest
- Feed-Forward Neural Network (MLP)

## Dataset

Freddie Mac Single-Family Loan-Level 2024 sample:

| File | Records | Columns | Content |
|---|---|---|---|
| `sample_orig_2024.txt` | 50,000 | 31 | Loan characteristics at origination |
| `sample_perf_2024.txt` | 952,865 | 35 | Monthly loan performance |

A loan is labelled **prepaid** if any of its performance records has `zero_balance_code = 1`. The resulting dataset has 50,000 loans: 40,823 not prepaid (81.65%) and 9,177 prepaid (18.35%), so the classes are imbalanced.

> The raw data files are not included in this repository. Download the Freddie Mac 2024 sample files from the Freddie Mac Single-Family Loan-Level Dataset page and place them as shown in the project structure below.

## Methodology

- **Features:** 28 origination-time features (14 numerical, 14 categorical); loan identifiers excluded.
- **Preprocessing:** median imputation and standard scaling for numerical features; most-frequent imputation and one-hot encoding for categorical features, all inside a scikit-learn `Pipeline` fitted on training data only.
- **Split:** stratified 60% train / 20% validation / 20% test (`random_state = 42`).
- **Class imbalance:** `class_weight="balanced"` for Logistic Regression, SVM and Random Forest.
- **Threshold tuning:** each model's decision threshold is chosen to maximise F1 on the **validation set only**.
- **Model selection:** the final model is the one with the highest **validation F1**; the test set is used only once, for final evaluation.
- **SVM scoring:** `decision_function()` is used instead of slow probability calibration (`probability=False`).

## Results

Validation F1 and test-set performance (thresholds fixed from validation):

| Model | Threshold | Val. F1 | Accuracy | Precision | Recall | Test F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.5461 | 0.4281 | 0.7095 | 0.3389 | 0.6131 | 0.4365 | 0.7375 |
| **RBF SVM (selected)** | **0.3167** | **0.4304** | **0.7475** | **0.3722** | **0.5477** | **0.4432** | **0.7393** |
| Random Forest | 0.4762 | 0.4194 | 0.7529 | 0.3725 | 0.5063 | 0.4292 | 0.7345 |
| Neural Network | 0.2230 | 0.4294 | 0.7327 | 0.3554 | 0.5613 | 0.4352 | 0.7349 |

The selected **RBF SVM** correctly identifies 1,005 of the 1,835 prepaid loans in the test set (recall 54.77%).

Confusion matrix on the test set:

| | Predicted Not Prepaid | Predicted Prepaid |
|---|---|---|
| **Actual Not Prepaid** | 6,470 | 1,695 |
| **Actual Prepaid** | 830 | 1,005 |

The four models perform very similarly, so the margin between them is small. Accuracy alone is misleading here: always predicting "not prepaid" would give about 81.65% accuracy while detecting no prepayments, which is why F1 and ROC-AUC are the primary metrics.

## Project Structure

```
.
├── app.py              # Streamlit application using the saved model
├── notebooks/          # Jupyter notebook with the full analysis
├── models/             # Saved model, threshold and metadata
├── results/            # Result outputs
├── documents/          # Project report (PDF)
├── data/               # (not tracked) raw Freddie Mac files
│   └── raw/sample_2024/
│       ├── sample_orig_2024.txt
│       └── sample_perf_2024.txt
├── .gitignore
└── README.md
```

Saved artefacts in `models/`:

- `mortgage_prepayment_final.pkl`: final trained pipeline (preprocessing + RBF SVM)
- `final_threshold.pkl`: selected threshold (0.3167)
- `final_model_metadata.pkl`: model name, threshold, dataset and selection metric

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/sanju819889/Mortgage-Prepayment-Prediction-ML.git
cd Mortgage-Prepayment-Prediction-ML
```

**2. Install dependencies** (Python 3.10+ recommended)

```bash
pip install numpy pandas matplotlib scikit-learn joblib streamlit jupyter
```

**3. Add the data**

Place the two Freddie Mac 2024 sample files in `data/raw/sample_2024/`.

**4. Run the notebook**

```bash
jupyter notebook notebooks/
```

Open the analysis notebook and run all cells. It trains the four models, selects the best one on validation F1, evaluates it on the test set, saves the model to `models/` and verifies that the reloaded model reproduces the test F1 of 0.4432.

**5. Launch the Streamlit app**

```bash
streamlit run app.py
```

## Limitations and Future Work

- Uses a 50,000-loan sample from a single vintage, with mainly origination-time features.
- The target is binary and does not model *when* a prepayment happens, so loans that have not yet prepaid are treated as non-prepaid.
- Possible extensions: macroeconomic and temporal variables (for example, market rate versus loan rate), testing on later loan vintages, systematic hyperparameter search, calibrated probabilities, and a survival (time-to-event) formulation.

## Reference Note

This project is an **extension** of the reference paper rather than a direct accuracy comparison, because the dataset period, sample and experimental setup differ. The additions are the newer 2024 sample, a Random Forest comparison, explicit class-imbalance handling, validation-based threshold tuning, leakage-safe model selection and deployment-ready model saving.
