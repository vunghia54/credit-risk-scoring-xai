# Credit Risk Scoring & Explainable AI: CV and Portfolio Descriptions

Use the versions below as alternatives for different application formats. Results describe the existing portfolio implementation and its internal evaluation, not a deployed lending system. Average Precision is abbreviated as AP.

## One-Line Project Summaries

**A. Data Scientist:** Developed a credit-risk workflow on 150,000 labeled observations, achieving internal-test ROC-AUC of 0.8697 and AP of 0.4161 with a frozen XGBoost model.

**B. Machine Learning Engineer:** Built a reproducible credit-risk pipeline with verified model serialization, a local FastAPI inference service, and 410 passing automated tests.

**C. Data Analyst / Risk Analytics:** Analyzed 150,000 credit-risk observations and documented data-quality decisions, precision-recall trade-offs, and SHAP explanations of serious-delinquency predictions.

## CV Bullets

### A. Compact Version

- Designed group-aware splits and leakage-safe preprocessing for 150,000 credit-risk observations, keeping identical feature vectors within the same partition.
- Benchmarked linear and tree models with imbalance, threshold, and calibration experiments; achieved frozen XGBoost internal-test ROC-AUC of 0.8697 and AP of 0.4161.
- Built global/local SHAP explanations and a local FastAPI inference service backed by verified serialization and 410 passing automated tests across the project.

### B. Standard Version

- Designed reproducible group-aware splits and leakage-safe preprocessing for 150,000 credit-risk observations, preventing identical feature vectors from crossing partitions.
- Benchmarked Logistic Regression, Random Forest, XGBoost, and LightGBM; evaluated class weighting, SMOTE, thresholds, and probability calibration using development data.
- Evaluated frozen XGBoost on the internal holdout, achieving ROC-AUC of 0.8697 and AP of 0.4161; documented trade-offs at a validation-selected operating threshold.
- Implemented global/local SHAP explanations and a local FastAPI inference service, with verified serialization and 410 passing automated tests across the workflow.

### C. Technical Version

- Implemented deterministic feature groups and approximately 70/15/15 GroupShuffleSplit partitions for 150,000 observations, verifying zero group and source-row overlap.
- Built training-fitted preprocessing and group-aware cross-validation for Logistic Regression, Random Forest, XGBoost, and LightGBM, confining SMOTE to fitting data.
- Evaluated targeted tuning, thresholds, and calibration; froze XGBoost Baseline / XGB-01 without calibration and thresholds 0.19/0.50 before final evaluation.
- Validated the frozen model on the internal test set: ROC-AUC 0.8697, AP 0.4161, and Brier score 0.0477; explained validation predictions with global/local SHAP.
- Integrated SHA-256-verified pipeline loading into a stateless local FastAPI service with strict input validation, prediction parity checks, and 410 passing project tests.

## Recommended CV Version

Use the Standard Version for a general application: it balances analysis, evaluation, explainability, and implementation without overstating deployment maturity.

- Designed reproducible group-aware splits and leakage-safe preprocessing for 150,000 credit-risk observations, preventing identical feature vectors from crossing partitions.
- Benchmarked Logistic Regression, Random Forest, XGBoost, and LightGBM; evaluated class weighting, SMOTE, thresholds, and probability calibration using development data.
- Evaluated frozen XGBoost on the internal holdout, achieving ROC-AUC of 0.8697 and AP of 0.4161; documented trade-offs at a validation-selected operating threshold.
- Implemented global/local SHAP explanations and a local FastAPI inference service, with verified serialization and 410 passing automated tests across the workflow.

## Short Portfolio Description

Developed a credit-risk scoring portfolio project using 150,000 labeled Give Me Some Credit observations to predict serious delinquency. The workflow combines group-aware splitting, training-fitted preprocessing, and comparisons of Logistic Regression, Random Forest, XGBoost, and LightGBM. Class weighting, SMOTE, threshold analysis, and calibration experiments informed development decisions. The frozen XGBoost model achieved internal-test ROC-AUC of 0.8697 and Average Precision of 0.4161. Global/local SHAP explains validation predictions, while a local FastAPI service loads the verified model artifact for inference. The project includes 410 passing automated tests and documents evaluation limitations; it is an educational portfolio, not a production lending system.

## Long Portfolio Description

This project explores serious-delinquency prediction and the trade-off between identifying higher-risk observations and generating false positives. It uses 150,000 labeled observations from Give Me Some Credit, with SeriousDlqin2yrs representing delinquency of 90 days past due or worse within two years.

The workflow groups identical raw feature vectors before an approximately 70/15/15 train/validation/test split. Preprocessing statistics are learned from training or fitting folds, and group-aware cross-validation supports comparisons of Logistic Regression, Random Forest, XGBoost, and LightGBM. Experiments cover class weighting, SMOTE, targeted tuning, threshold selection, and probability calibration.

XGBoost Baseline / XGB-01 was frozen without calibration before final internal test evaluation, achieving ROC-AUC of 0.8697, Average Precision of 0.4161, and Brier score of 0.0477. At the development threshold of 0.19, selected on validation, test precision was 0.4034, recall 0.5319, and F1 0.4588. The default audit threshold remains 0.50; neither threshold was tuned on test.

Global and local SHAP explain validation predictions in raw-margin/log-odds space. A stateless local FastAPI service loads the serialized pipeline after SHA-256 verification, with strict input schemas and prediction parity checks. The project has 410 passing automated tests.

The evaluation is internal, not external or temporal. EDA preceded the final split lock, so the analyst had prior visibility into the full labeled dataset. SHAP does not establish causality, fairness was not assessed, and the scores are not regulatory probabilities of default. The API is a local portfolio demonstration, and the model does not make lending decisions.

## LinkedIn Project Description

Developed a Python credit-risk scoring project using 150,000 labeled Give Me Some Credit observations to predict serious delinquency. The work connects data-quality analysis, model evaluation, explainability, and a local inference API.

- Designed group-aware splits and training-fitted preprocessing to prevent duplicate-vector leakage across partitions.
- Compared Logistic Regression, Random Forest, XGBoost, and LightGBM, including imbalance, threshold, and calibration experiments.
- Evaluated frozen XGBoost on an internal holdout: ROC-AUC 0.8697 and Average Precision 0.4161, with decisions fixed before final testing.
- Implemented global/local SHAP, verified pipeline serialization, and a FastAPI service; the project includes 410 passing automated tests.

This is an educational portfolio with internal validation, documented EDA visibility limitations, and no production lending claims. SHAP explains model behavior rather than causality; API outputs are risk scores, not credit approval or rejection decisions.

## GitHub Repository Description

### A. Short Description

Credit-risk scoring with group-aware evaluation, XGBoost, SHAP, and a tested local FastAPI inference service.

### B. About / Portfolio Tagline

A reproducible serious-delinquency prediction portfolio covering model comparison, threshold/calibration analysis, and SHAP explanations. Frozen XGBoost achieved internal-test ROC-AUC of 0.8697 and AP of 0.4161, with a local FastAPI inference service.

## Skills / Keywords

| Area | Demonstrated skills and technologies |
| --- | --- |
| Machine Learning | Python, scikit-learn, Logistic Regression, Random Forest, XGBoost, LightGBM, targeted hyperparameter tuning |
| Data Analysis | pandas, NumPy, Matplotlib, EDA, missing-value analysis, duplicate analysis, data-quality documentation |
| Risk Modeling | Serious-delinquency prediction, class imbalance, class weighting, imbalanced-learn, SMOTE, precision-recall trade-offs |
| Explainable AI | SHAP, global feature importance, local explanations, raw-margin/log-odds interpretation, additivity checks |
| Model Evaluation | ROC-AUC, Average Precision, Gini, Brier score, log loss, ECE, threshold analysis, sigmoid/isotonic calibration analysis |
| API / Model Delivery | FastAPI, Pydantic, Uvicorn, joblib serialization, SHA-256 integrity checks, stateless local inference, input validation |
| Testing / Reproducibility | pytest, FastAPI TestClient, prediction parity, deterministic group-aware splits, group-aware CV, pinned dependencies, Git |

## What to Emphasize by Role

| Role | Emphasize | De-emphasize |
| --- | --- | --- |
| Data Analyst | EDA, data-quality decisions, visual communication, metric interpretation, precision-recall trade-offs | Estimator parameter lists and serialization internals |
| Data Scientist | Group-aware evaluation, model comparisons, imbalance experiments, tuning, calibration, SHAP, holdout limitations | Endpoint implementation details |
| Machine Learning Engineer | Reusable preprocessing, frozen pipeline serialization, integrity checks, strict API schemas, parity tests, reproducibility | Exhaustive EDA findings and individual plot descriptions |
| Risk / Credit Analytics | Serious-delinquency definition, score ranking and calibration, threshold trade-offs, explainability, governance limitations | Infrastructure details and leaderboard-style comparisons |

De-emphasis changes presentation, not the facts or limitations. Keep the evaluation labeled as internal for every role.

## Claims to Avoid

| Claim to avoid | Why it is unsupported |
| --- | --- |
| Production-ready credit decision engine | The implementation is an educational portfolio, without operational lending validation or governance approval. |
| Regulatory PD model | Internal probability evaluation does not establish regulatory or production PD validity. |
| Bank-approved model | No bank approval or institutional validation is documented. |
| Causal SHAP explanations | SHAP explains model predictions; it does not establish causal relationships. |
| External validation | Results come from an internal split of the same dataset, not an independent external or temporal sample. |
| Fairness validated | No protected-group fairness assessment was performed; age is present, but attributes such as sex and race are absent. |
| Deployed to cloud | The repository documents local execution, not a cloud deployment. |
| Production API | The FastAPI service is a local demonstration without authentication, rate limiting, or production validation. |
| Reduced defaults, losses, or review costs | No measured business intervention, financial cost matrix, or operational impact study exists. |
| Bank-optimal threshold | Threshold 0.19 follows a validation recall/precision rule, not a validated lending policy or business-cost optimum. |

## Metric Reference and Evidence

Use the rounded values above for concise application text. The table below preserves six-decimal reporting precision from existing artifacts; it does not represent a new evaluation.

| Item | Verified value / scope |
| --- | --- |
| Final model | XGBoost Baseline / XGB-01 |
| Calibration | none |
| Development operating threshold | 0.19, selected on validation before final test |
| Default audit threshold | 0.50 |
| Internal-test ROC-AUC | 0.869663 |
| Internal-test Average Precision | 0.416082 |
| Internal-test Gini | 0.739325 |
| Internal-test Brier score | 0.047727 |
| Internal-test log loss | 0.173379 |
| Internal-test ECE | 0.005445 |
| Internal-test precision at 0.19 | 0.403366 |
| Internal-test recall at 0.19 | 0.531944 |
| Internal-test F1 at 0.19 | 0.458817 |
| Automated tests | 410 passing at the completed repository audit |

Precision, recall, and F1 above are fractions, not percentages. Test results are reported only as final evaluation outcomes; model selection, calibration, and operating-threshold decisions used development evidence. The consumed test set was not used for SHAP, serialization verification, or API parity development. No business impact or deployment outcome is inferred from these metrics.

Sources: [project overview](../README.md), [final model selection](final_model_selection.md), [final internal test evaluation](final_test_evaluation.md), [full-precision test metrics](final_test_results.json), [global SHAP](shap_global_explainability.md), [local SHAP](shap_local_explainability.md), [serialization audit](model_serialization.md), and [API implementation](api_implementation.md).
