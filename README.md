# ML-Powered Financial Risk Scoring System

An end-to-end machine learning project for borrower credit risk classification, interpretable risk assessment, and portfolio-level financial risk analysis.

## Overview

The **ML-Powered Financial Risk Scoring System** uses supervised machine learning to evaluate borrower credit risk based on historical lending data. The project explores multiple classification algorithms, including Logistic Regression, Random Forest, and XGBoost, to distinguish between fully repaid loans and charged-off loans.

The system combines data preprocessing, model evaluation, explainable AI, and credit risk scorecard development to support data-driven lending decisions. An interactive Streamlit dashboard is intended to make borrower-level predictions and portfolio risk visualizations accessible through a user-friendly interface.

## Key Features

- **Credit Risk Classification:** Predict borrower loan outcomes using Logistic Regression, Random Forest, and XGBoost.
- **Data Preprocessing:** Clean and standardize lending data, handle missing values, process numerical and categorical variables, and remove potential target leakage.
- **Model Evaluation:** Compare models using classification metrics, ROC-AUC, ROC curves, confusion matrices, and precision-recall analysis.
- **Class Imbalance Handling:** Explore SMOTE-based oversampling to improve learning from imbalanced loan outcomes.
- **Explainable AI:** Use SHAP to interpret model predictions and identify features influencing borrower risk.
- **Credit Risk Scorecard:** Explore Weight of Evidence (WoE) transformations, probability calibration, and risk score generation.
- **Threshold Optimization:** Select classification thresholds based on precision, recall, and F1-score requirements.
- **Portfolio Risk Analysis:** Analyze score bands, expected loss estimates, and portfolio-level risk indicators.
- **Interactive Visualization:** Use Streamlit to present risk scores, model outputs, and portfolio analytics.

## Machine Learning Models

| Model | Purpose |
|---|---|
| Logistic Regression | Establish an interpretable baseline for credit risk classification |
| Random Forest | Capture nonlinear relationships and interactions between borrower characteristics |
| XGBoost | Apply gradient-boosted decision trees for predictive risk classification |

The models are evaluated on their ability to distinguish between two loan outcomes:

- **Fully Paid:** The loan was repaid.
- **Charged Off:** The loan was charged off.

The project also explores probability calibration and threshold selection to make classification decisions more suitable for credit risk assessment.

## Explainability with SHAP

SHAP (SHapley Additive exPlanations) provides a framework for interpreting model predictions.

The explainability component is designed to help understand:

- Which borrower attributes influence risk predictions.
- How individual features contribute to a model's output.
- Which features have the greatest overall influence on predictions.
- How model behavior can be communicated to lending and risk analysis stakeholders.

This improves the transparency of model-assisted credit risk assessment.

## Credit Risk Scoring and Portfolio Analytics

Beyond binary classification, the project explores transforming predicted default probabilities into credit risk scores.

The scoring workflow includes:

1. Transforming borrower features using Weight of Evidence (WoE).
2. Training and evaluating a Logistic Regression scorecard.
3. Calibrating predicted probabilities.
4. Converting predicted risk probabilities into numerical scores.
5. Grouping borrowers into score bands.
6. Estimating expected credit losses using probability of default (PD), loss given default (LGD), and exposure at default (EAD).
7. Evaluating operating thresholds against selected precision and recall requirements.

These components demonstrate how machine learning predictions can be translated into interpretable risk indicators for portfolio monitoring.

## Technology Stack

- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost, imbalanced-learn
- **Explainable AI:** SHAP
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Interactive Dashboard:** Streamlit
- **Development Environment:** Jupyter Notebook / Google Colab

## Project Structure

```text
ML-Powered-Financial-Risk-Scoring/
│
├── Copy_of_AI_Powered_Financial_Risk_Scoring_System.ipynb
├── app.py                 # Streamlit dashboard, if included
├── requirements.txt       # Python dependencies
├── data/                  # Lending dataset (not included)
├── models/                # Saved model artifacts, if exported
└── README.md
```

*The additional files and directories above are suggested repository organization; include them only if they are part of your repository.*

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ML-Powered-Financial-Risk-Scoring
```

### 2. Install dependencies

Create a virtual environment and install the required libraries.

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap streamlit plotly joblib
```

### 3. Prepare the dataset

The notebook uses a LendingClub accepted-loans dataset and references a cleaned subset named `clean_lendingclub_subset.csv` in later stages.

Obtain the dataset from an appropriate source, place the required files in the expected locations, and update the notebook's file paths as necessary. The original dataset is not included in this repository.

### 4. Run the notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `Copy_of_AI_Powered_Financial_Risk_Scoring_System.ipynb` and execute the cells in sequence to explore data preprocessing, model development, evaluation, and credit risk scoring.

### 5. Launch the dashboard

If the repository includes a configured `app.py`:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal to interact with the dashboard.

*The notebook contains Streamlit setup commands, but the dashboard requires a compatible application script and the necessary model artifacts or inference pipeline.*

## Evaluation Metrics

The project uses several complementary metrics to assess model performance:

- **ROC-AUC:** Measures the model's ability to rank borrowers by risk.
- **Precision:** Measures the proportion of predicted high-risk cases that are actually charged off.
- **Recall:** Measures the proportion of charged-off loans identified by the model.
- **F1-score:** Balances precision and recall.
- **Confusion Matrix:** Summarizes correct and incorrect classifications.
- **Calibration Analysis:** Examines how predicted probabilities align with observed outcomes.

No single metric fully captures lending risk. Model selection and classification thresholds should reflect the intended lending objective and the relative costs of false positives and false negatives.

## Potential Applications

- Borrower credit risk assessment.
- Loan portfolio segmentation.
- Explainable lending analytics.
- Credit risk scorecard development.
- Expected credit loss estimation.
- Portfolio risk monitoring and visualization.

## Limitations

This project is an educational and analytical implementation rather than a production-ready lending system. Historical lending data may contain sampling bias, changing economic conditions, and limitations in representing future borrowers.

Model outputs should be validated on appropriately separated datasets and monitored for calibration, data drift, fairness, and changes in loan performance. Expected loss estimates also depend on the assumptions used for LGD and EAD.

## Future Improvements

- Deploy a complete Streamlit application with borrower-level risk predictions.
- Integrate interactive SHAP explanations into the dashboard.
- Add automated model comparison and hyperparameter optimization.
- Improve probability calibration and scorecard validation.
- Introduce fairness and model-drift monitoring.
- Develop a reproducible model-training and deployment pipeline.

## Disclaimer

This project is intended for educational and research purposes. Predictions are not financial advice and should not be used as the sole basis for real-world lending decisions.

## Author

**Prayag Raghunandan**

