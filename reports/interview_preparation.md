# Credit Risk Scoring & Explainable AI: Interview Preparation

Practice guide for the completed portfolio. Spoken scripts use rounded numbers; detailed tables retain recorded precision. Every performance result is labeled by its evaluation partition. Timing labels are rehearsal targets, not measured delivery times; shorten transitions if needed rather than rushing the technical claims.

## 30-Second Elevator Pitch

I developed a credit-risk project using 150,000 Give Me Some Credit observations to predict serious delinquency. I focused on leakage-safe preprocessing and group-aware evaluation, then compared Logistic Regression, Random Forest, XGBoost, and LightGBM. The frozen XGBoost model achieved internal-test ROC-AUC of 0.8697 and Average Precision of 0.4161. I added SHAP explanations and a local FastAPI inference service. The main lesson was that credible evaluation requires more than a good score: thresholds, calibration, reproducibility, and clearly disclosed limitations matter too.

## 2-Minute Project Explanation

The project predicts serious delinquency using Give Me Some Credit. The target is whether someone experienced delinquency of 90 days past due or worse within two years, rather than bankruptcy. There are 150,000 labeled observations, and only 6.684% are positive, so accuracy alone would hide weak minority-class performance.

The data also contains missing income, unusual delinquency values, and repeated feature vectors. I kept identical raw feature vectors in the same partition and learned preprocessing statistics only from training or fitting folds. That gave me a reproducible evaluation design without pretending the groups identify individual borrowers.

I compared Logistic Regression, Random Forest, XGBoost, and LightGBM, including class weighting, SMOTE, and targeted tuning. I retained XGBoost Baseline because the tuning gains were small and the development evidence supported a simpler fixed candidate. On the final internal test, it achieved ROC-AUC of 0.8697 and Average Precision of 0.4161.

Before opening test, I selected threshold 0.19 on validation by requiring at least 50% recall and then maximizing precision. I kept 0.50 as an audit reference. Calibration experiments did not justify adding a calibration layer to the final model, so it remained uncalibrated.

I used global and local SHAP on validation observations to explain model behavior, then exposed the frozen pipeline through a local FastAPI service with input validation and artifact integrity checks. The project passed 410 automated tests at the repository audit.

The main limitation is that EDA preceded the final split lock, so the analyst was not completely blind to the eventual holdout. This is an internal portfolio evaluation, not external validation or a production lending system.

## 5-Minute Technical Presentation

### Opening

This project asks how to build a credible serious-delinquency scoring workflow, not just a high-scoring classifier. I will walk through the data decisions, development evidence, frozen evaluation, and inference boundary.

### Data

Give Me Some Credit provides 150,000 labeled observations, with 6.684% positive labels and ten raw predictors. That imbalance and the data's recording conventions made quality checks the first modeling concern.

### Data Quality

I found missing income and dependents, zero ages, undocumented delinquency values 96 and 98, and extreme ratios. I documented explicit handling rules without changing raw data, then examined repeated predictors before splitting.

### Split Strategy

There were 646 excess duplicate feature vectors across 354 repeated groups; a standard stratified comparator dispersed 197 groups containing 605 observations across partitions. Two-stage GroupShuffleSplit kept feature-identical rows together in an approximately 70/15/15 split, giving preprocessing a defined training boundary.

### Preprocessing

I fitted medians only on training or fitting folds, preserved observed zero income, and added missingness and special-value indicators. Linear models used numeric scaling, while tree models used the same quality rules without scaling, allowing consistent model comparisons.

### Model Experiments

I compared Logistic Regression, Random Forest, XGBoost, and LightGBM using group-aware training CV and validation metrics. Boosting offered stronger development ranking than the linear baseline, but I also tested whether imbalance treatments changed the operating trade-off.

### Class Imbalance

Class weighting and SMOTE improved Logistic recall but increased false positives. SMOTE had slightly higher validation recall and lower precision and F1 than Balanced Logistic, so I retained it as a benchmark rather than assuming resampling was necessary.

### Final Model Selection

I froze XGBoost Baseline / XGB-01: tuned XGBoost added only a small AP gain, and LightGBM was a close alternative rather than a poor performer. Training CV, validation behavior, and simplicity informed that choice before I settled the operating threshold.

### Threshold

On validation, I selected 0.19 by requiring recall of at least 50%, then maximizing precision, with predeclared tie-breaks. I retained 0.50 for audit comparison; neither is a bank-approved policy, and ranking alone does not establish probability quality.

### Calibration

I compared uncalibrated, sigmoid, and isotonic variants using training-only calibration folds. Sigmoid worsened XGBoost probability metrics and isotonic offered very small Brier/log-loss gains, so I froze calibration as none before opening test.

### Final Test

The frozen model achieved internal-test ROC-AUC of 0.8697 and AP of 0.4161; at threshold 0.19, precision was 0.4034 and recall was 0.5319. That holdout was consumed in STEP 15 and did not guide later model changes or explainability work.

### SHAP

Global and local SHAP used validation observations, with revolving utilization the highest-ranked global feature. Contributions reconstruct XGBoost's raw margin in log-odds space, explaining predictions without establishing causality; the next step was preserving the same pipeline for inference.

### Serialization/API

I verified serialization through validation load-back parity and SHA-256 checks, then built a stateless local FastAPI service accepting ten raw features. The project passed 410 automated tests, but these engineering checks do not establish production readiness.

### Limitations

EDA saw the full labeled dataset before the split was locked, and the evaluation is neither external nor temporal. Fairness was not assessed, and the API lacks production controls such as authentication and rate limiting, which bounds what I can claim.

### Closing

The strongest outcome is a reproducible, documented chain from raw inputs to evaluated scores and explanations. I learned to separate ranking, operating decisions, probability quality, and serving correctness instead of treating one metric as proof of success.

## Interview Storyline

Problem → Risk → Data → Leakage → Baseline → Boosting → Threshold → Calibration → Test → Explainability → API → Limitations

| Step | Key message | Fact to remember |
| --- | --- | --- |
| Problem | Predict the dataset's serious-delinquency event. | Target: `SeriousDlqin2yrs`; two-year window |
| Risk | Minority-class performance matters more than majority-class accuracy. | Full-data positive prevalence: 6.684% |
| Data | Understand recording issues before making transformations. | 150,000 observations; ten raw predictors |
| Leakage | Keep identical raw predictors in the same partition. | 646 excess duplicate feature vectors |
| Baseline | Establish an interpretable linear reference first. | Logistic Baseline validation recall: 0.163735 at 0.50 |
| Boosting | Compare development evidence, not just the biggest validation number. | Frozen candidate: XGB-01 |
| Threshold | Choose an operating point using validation only. | 0.19: recall >= 50%, then highest precision |
| Calibration | Assess probability reliability separately from ranking. | Final calibration: none |
| Test | Evaluate frozen choices and stop using test for decisions. | Internal-test ROC-AUC/AP: 0.8697 / 0.4161 |
| Explainability | Explain validation predictions without claiming causality. | SHAP uses raw-margin/log-odds output |
| API | Serve the frozen pipeline without training in the request path. | Ten required raw inputs |
| Limitations | State what the evidence does not establish. | EDA preceded the final split lock |

## Core Technical Questions

### Q01. What is the target, and how imbalanced is it?

**Short answer:** `SeriousDlqin2yrs` indicates serious delinquency, with 10,026 positives among 150,000 observations: 6.684%.

**Deeper answer:** The documented event is 90 days past due delinquency or worse within two years. It is not bankruptcy. The minority prevalence makes positive-class precision and recall essential, and the internal test prevalence is about 6.6% rather than exactly the full-data rate.

**Possible follow-up:** How would a different event definition or population change the interpretation of the score?

### Q02. Why not just use accuracy?

**Short answer:** Predicting the majority class can produce high accuracy while missing every positive event.

**Deeper answer:** With positives around 6.6%, accuracy does not tell me whether high-risk observations are identified. I report ROC-AUC and AP for ranking, then precision, recall, F1, and confusion matrices at frozen thresholds. No business benefit follows from accuracy alone.

**Possible follow-up:** Could a lower-accuracy classifier be more useful for a recall-focused screening task?

### Q03. How do ROC-AUC and Average Precision differ?

**Short answer:** ROC-AUC summarizes discrimination through true- and false-positive rates; AP emphasizes precision as positive cases are retrieved.

**Deeper answer:** ROC-AUC has a pairwise ranking interpretation, with ties receiving partial credit. AP weights precision by recall increments and is sensitive to prevalence; it is not trapezoidal PR area. For the final internal test, ROC-AUC was 0.869663 and AP was 0.416082. Neither is classification accuracy.

**Possible follow-up:** Why should AP comparisons account for the evaluated population's positive prevalence?

### Q04. Why did you use a group-aware split?

**Short answer:** I wanted to prevent identical raw predictors from appearing in different evaluation partitions.

**Deeper answer:** Excluding target and index, there were 646 duplicate feature vectors after first occurrence, distributed across 354 multi-observation groups. The recorded standard stratified comparator dispersed 197 distinct groups containing 605 observations across partitions. Grouping all ten raw predictors, including matching missingness, removed that overlap. Equal predictors do not prove equal borrower identity: this is protection against duplicate-vector leakage, not identity-level deduplication or proof of a measured performance inflation.

**Possible follow-up:** What leakage risks remain if near-duplicates or repeated borrowers cannot be identified?

### Q05. Why not a standard stratified split?

**Short answer:** Stratification preserves class proportions but does not keep feature-identical groups together.

**Deeper answer:** The chosen holdout method uses two GroupShuffleSplit stages with seed 42. It is not explicitly stratified; group and row disjointness, approximate partition sizes, and class-rate bounds are checked separately. Training CV uses StratifiedGroupKFold, which is a different procedure. The standard stratified split was an integrity comparator, not the modeling split.

**Possible follow-up:** How would you handle a group split that failed the predeclared class-balance checks?

### Q06. Why retain duplicate observations instead of deleting them?

**Short answer:** Matching feature vectors are not enough evidence that records are accidental duplicate borrowers.

**Deeper answer:** Some identical predictors have different labels, and the source index is not a documented customer identifier. Removing rows could alter the observed population without a defensible identity rule. I retained observations and isolated their raw feature groups across partitions; that does not solve every dependence or data-provenance issue.

**Possible follow-up:** What source-system evidence would justify deduplication?

### Q07. How is preprocessing leakage controlled?

**Short answer:** Split first, then fit preprocessing statistics only on training or the current fitting fold.

**Deeper answer:** Missing income and dependents receive training-fitted medians plus indicators. A combined delinquency special-value indicator is also created, giving ten numeric inputs and three indicators. Linear pipelines scale the numeric features; tree pipelines do not scale. Validation and test use fitted transformations unchanged, and raw files stay intact.

**Possible follow-up:** Why is fitting an imputer before cross-validation a form of leakage?

### Q08. Why treat age equal to zero as missing?

**Short answer:** Zero age is implausible for this credit context, so the documented rule treats it as unavailable in the working representation.

**Deeper answer:** The frozen preprocessor replaces zero age with missing and applies a training-fitted median. It does not delete the record or alter raw data. High observed ages are retained without an arbitrary age cap; this is an explicit modeling policy, not recovery of a person's true age.

**Possible follow-up:** What would change if authoritative source documentation explained the zero value?

### Q09. Why distinguish missing income from income equal to zero?

**Short answer:** Unobserved income and observed zero income convey different information.

**Deeper answer:** Missing `MonthlyIncome` activates its missingness flag and is imputed using the training median. Zero remains zero with that flag off. `NumberOfDependents` can also be missing and follows its own indicator/imputation rule. The API preserves these distinctions rather than converting all falsy values into nulls.

**Possible follow-up:** What test would catch a zero-to-null conversion bug in serving?

### Q10. What do delinquency values 96 and 98 mean?

**Short answer:** Their meaning is undocumented; I do not call them confirmed error codes.

**Deeper answer:** They occur in the three delinquency-count fields. The pipeline first records whether any such value occurs, then treats affected values as missing in a working copy and applies medians fitted without those special values. This avoids interpreting undocumented codes as ordinary ordinal counts while retaining a signal. The combined flag loses distinctions between codes and columns.

**Possible follow-up:** Could that flag capture a source-system convention rather than transferable risk behavior?

### Q11. Why keep extreme ratios and counts?

**Short answer:** I had no authoritative business rule justifying an arbitrary cap.

**Deeper answer:** Removing or winsorizing observations could discard genuine signal. Tree models can represent nonlinear splits without requiring those transformations, although that does not make them immune to unreliable inputs or distribution shift. Any later representation change would need a new development protocol, not selection using the consumed test.

**Possible follow-up:** What evidence would support a log transform or domain-based cap?

### Q12. What did class weighting accomplish?

**Short answer:** It increased Logistic recall at the default threshold, with more false positives.

**Deeper answer:** At validation threshold 0.50, unweighted Logistic recall was 0.163735. Balanced Logistic reached recall 0.618403 and precision 0.249522. Weighting changes the fitting objective rather than generating observations; the resulting scores also needed separate calibration analysis. Higher recall alone does not establish a better overall strategy.

**Possible follow-up:** Why can class weighting change probability calibration?

### Q13. Why was SMOTE not the final strategy?

**Short answer:** It was a useful benchmark, but its extra recall came with lower precision and F1 than Balanced Logistic.

**Deeper answer:** The following are validation results at 0.50, not test comparisons:

| Logistic strategy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Balanced | 0.249522 | 0.618403 | 0.355573 |
| SMOTE | 0.237168 | 0.634641 | 0.345297 |

SMOTE ran only inside fitting data. Interpolating transformed counts and indicators can also create fractional profiles that do not correspond to actual applicants. Boosting offered stronger development ranking and relevant threshold trade-offs, so I did not select SMOTE; that is not a claim that SMOTE is generally bad.

**Possible follow-up:** How would categorical or constrained features affect your choice of resampling method?

### Q14. Why include Random Forest?

**Short answer:** It provided a nonlinear tree benchmark alongside the linear reference and boosting candidates.

**Deeper answer:** Random Forest averages randomized trees, whereas boosting builds an additive model sequentially. Its role was to test a different model family under the same evaluation discipline. The recorded development comparisons favored boosting; I did not compare alternative models on the final test or claim one family is universally superior.

**Possible follow-up:** What kinds of data properties might change that comparison?

### Q15. Why XGBoost Baseline rather than tuned XGBoost?

**Short answer:** The tuning gain was small, validation ROC-AUC decreased slightly, and the fixed baseline remained a defensible simpler choice.

**Deeper answer:** These are development metrics:

| XGBoost candidate | Train CV AP | Validation AP | Validation ROC-AUC |
| --- | ---: | ---: | ---: |
| Baseline / XGB-01 | 0.398880 | 0.381284 | 0.863666 |
| Tuned / XGB-07 | 0.400384 | 0.382030 | 0.863488 |

XGB-07 changed row/column subsampling, not the tree count or depth, so I do not imply a much larger estimator. The development decision preferred the existing baseline over an additional tuned configuration for marginal gains. No statistical significance or universal superiority was established.

**Possible follow-up:** What evidence would make a small metric improvement worth accepting?

### Q16. Why not LightGBM?

**Short answer:** It was a close alternative, but XGB-01 had slightly stronger training CV AP, the primary development ranking measure.

**Deeper answer:** LGBM-02 had train CV AP 0.397801, validation AP 0.381685, and validation ROC-AUC 0.864139. XGB-01 had 0.398880, 0.381284, and 0.863666 respectively. LightGBM was marginally ahead on validation; those small differences did not establish a decisive winner. I retained the fixed XGBoost candidate using the combined development evidence, not because LightGBM performed poorly.

**Possible follow-up:** Would latency or operational constraints change selection if they were measured?

### Q17. How was tuning isolated from test?

**Short answer:** Candidate hyperparameters were selected using training-only group-aware cross-validation.

**Deeper answer:** The targeted search used shared five-fold training splits and ranked configurations primarily by mean CV AP. Validation supplied subsequent development comparisons; it was not the tuning search objective, and test was not used for selection. Preprocessing was refitted inside each fitting fold. This is a targeted search, not proof that every useful configuration was explored.

**Possible follow-up:** What is the difference between tuning on training CV and repeatedly making decisions on validation?

### Q18. Why threshold 0.19?

**Short answer:** It was the validation operating point with highest precision among evaluated thresholds reaching at least 50% recall.

**Deeper answer:** Ties used higher F1, then higher threshold; the prediction rule is `score >= threshold`. The operating point was frozen before the internal test:

| Partition at 0.19 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Validation | 0.372303 | 0.502030 | 0.427542 |
| Final internal test | 0.403366 | 0.531944 | 0.458817 |

The test row reports the frozen policy's outcome, not a new selection. There was no supplied cost matrix, so I do not call 0.19 bank-optimal, cost-optimal, or regulatory policy.

**Possible follow-up:** How would explicit false-positive and false-negative costs change the decision process?

### Q19. Why retain 0.50 too?

**Short answer:** It is a default audit/reference point, not the selected development operating point.

**Deeper answer:** On final test, 0.50 produced precision 0.602592, recall 0.187626, and F1 0.286154. It illustrates the more conservative precision-recall trade-off relative to 0.19. Both were specified before final testing and remain unchanged; neither output flag instructs a lender to approve or decline credit.

**Possible follow-up:** Why is a probability threshold not itself a lending policy?

### Q20. What is calibration?

**Short answer:** Calibration asks whether predicted probabilities agree with observed event frequencies.

**Deeper answer:** A model may rank observations well while consistently overestimating their event probabilities. I examined reliability curves, Brier score, log loss, and ECE rather than treating ROC-AUC as probability validation. Sigmoid and isotonic alternatives were fitted using training folds and compared on validation.

**Possible follow-up:** Can ranking stay similar while probability quality changes substantially?

### Q21. Why leave final XGBoost uncalibrated?

**Short answer:** Sigmoid worsened probability metrics, while isotonic's small Brier/log-loss gains did not justify another calibration ensemble.

**Deeper answer:** The recorded XGBoost validation comparison was:

| Method | Brier score | Log loss | ECE |
| --- | ---: | ---: | ---: |
| Uncalibrated | 0.049105 | 0.177687 | 0.005417 |
| Sigmoid | 0.050795 | 0.189396 | 0.020427 |
| Isotonic | 0.049070 | 0.177558 | 0.003952 |

Isotonic's ECE improvement is visible, but the Brier/log-loss changes are very small; the decision weighed the whole picture. Calibrated alternatives were fold ensembles, so this was not an isolated comparison of score mappings alone. Uncalibrated does not mean regulatory PD validity, and no calibration layer was added to the frozen model.

**Possible follow-up:** Why might calibration performance change under population shift?

### Q22. What does Brier score tell you?

**Short answer:** It measures mean squared error between event probabilities and binary outcomes; lower is better.

**Deeper answer:** Final internal-test Brier score was 0.047727. It assesses overall probability prediction quality, reflecting both calibration and discrimination rather than isolating calibration alone. Its interpretation depends on the population and event prevalence, so I pair it with ranking metrics and reliability plots.

**Possible follow-up:** Why should Brier score be compared with a meaningful reference on the same population?

### Q23. What are ECE and log loss, and what are their limits?

**Short answer:** ECE aggregates probability-frequency gaps across bins; log loss penalizes wrong probability predictions, especially confident ones.

**Deeper answer:** The recorded test ECE is 0.005445 and log loss is 0.173379. This project requests ten quantile bins, removes duplicate edges, keeps identical scores together, and omits empty bins. ECE is binning-dependent and can hide local errors; neither a small ECE nor good mean probability proves calibration for every subgroup.

**Possible follow-up:** How could two binning strategies give different ECE values for identical predictions?

### Q24. Why evaluate the internal test only once?

**Short answer:** Repeated decisions based on test results turn test into development data.

**Deeper answer:** Model specification, preprocessing, calibration, and thresholds were frozen first, then evaluated in STEP 15. The test is now consumed, not unseen. Later SHAP and serialization checks use validation, and API parity uses toy inputs. Existing splitter integrity checks are distinct from using test predictions to choose models.

**Possible follow-up:** What would you do if the frozen model performed worse than expected on test?

### Q25. Why not refit on training plus validation before test?

**Short answer:** The purpose was to evaluate the training-only fitted development specification whose validation behavior had been frozen.

**Deeper answer:** Combining partitions would change the fitted preprocessing statistics and estimator, invalidating that exact development reference. The serialized pipeline is a deterministic reconstruction of the frozen training-only specification, verified on validation, not a claim to preserve the historical in-memory STEP 15 object. A future deployment retraining policy would require separate validation and governance.

**Possible follow-up:** How would you distinguish a frozen evaluation artifact from a later deployment artifact?

### Q26. What does SHAP show?

**Short answer:** It decomposes the model output into a reference value and feature contributions.

**Deeper answer:** Global importance is mean absolute SHAP on validation; local explanations describe particular validation predictions. The leading features were:

1. `RevolvingUtilizationOfUnsecuredLines`
2. `NumberOfTime30-59DaysPastDueNotWorse`
3. `NumberOfTimes90DaysLate`
4. `age`
5. `NumberOfTime60-89DaysPastDueNotWorse`

These explain the fitted model's associations, not causes of delinquency, legal adverse-action reasons, or proof of fairness.

**Possible follow-up:** How can correlated features affect attribution and importance rankings?

### Q27. Why are SHAP values not probability percentages?

**Short answer:** TreeExplainer used raw output, so its contributions are additive in raw-margin/log-odds space.

**Deeper answer:** `base value + sum(SHAP contributions)` approximately reconstructs the raw margin; applying sigmoid to that total reconstructs the probability. Sigmoid is nonlinear, so applying it independently to each contribution is not a probability decomposition. As an illustration, a SHAP value of +1.0 is not an increase of 100 percentage points. Additivity and probability reconciliation were checked.

**Possible follow-up:** Why can the same raw-margin contribution have different effects on total probability at different starting margins?

### Q28. How did you select the local SHAP cases?

**Short answer:** I used validation scores only, without target labels.

**Deeper answer:** The six cases are nearest the low, median, and high score quantiles, just below 0.19, at or just above 0.19, and nearest 0.50. The quantiles are the 5th, 50th, and 95th percentiles. Deterministic row-position tie-breaks and deduplication keep selections distinct. This avoids outcome-based cherry-picking, but six illustrative cases are not population-level validation.

**Possible follow-up:** Why might outcome-selected examples give a misleading impression of an explanation method?

### Q29. How did you serialize and verify the model?

**Short answer:** I persisted the complete fitted pipeline with joblib and verified validation predictions after loading it back.

**Deeper answer:** The artifact includes preprocessing and estimator state, with software versions and SHA-256 in metadata. The API verifies the exact bytes before deserialization, then checks the fitted structure, feature schema, and frozen settings; it rejects mismatches and loads once per lifespan. Only trusted joblib artifacts should be loaded: a checksum is not a signature, and compatible dependencies/custom classes are required.

**Possible follow-up:** What happens if an attacker can replace both the binary and its metadata?

### Q30. Why FastAPI, and what is the inference flow?

**Short answer:** It demonstrates a clear boundary between model development and local inference.

**Deeper answer:** A raw request passes schema validation, enters the frozen preprocessing pipeline, reaches `predict_proba`, and returns a probability plus frozen threshold flags. `GET /health`, `GET /model-info`, and `POST /predict` expose that contract. No training, resampling, calibration fitting, or test-data evaluation occurs in serving; requests are stateless.

**Possible follow-up:** Why serve the full pipeline rather than reimplementing preprocessing in the API?

### Q31. How is the API input contract validated?

**Short answer:** Pydantic requires exactly ten named raw fields, rejects extras, and preserves null/zero semantics.

**Deeper answer:** Income and dependents can be null while their keys remain required. Zero income remains observed zero; age zero and delinquency 96/98 pass to the frozen preprocessing rules. Invalid types, negative values, and nonfinite numbers are rejected. Sanitized errors avoid payload echo, and payloads are not logged by default; tests cover boundary equality and direct-artifact parity.

**Possible follow-up:** Does schema-valid input guarantee that a request resembles the training distribution?

### Q32. Is the API production-ready?

**Short answer:** No; it is a tested local portfolio inference service.

**Deeper answer:** The implementation demonstrates frozen pipeline loading, integrity checks, input validation, and inference parity. A production system would also need authentication/authorization, rate limiting, operational and drift monitoring, deployment infrastructure, privacy controls beyond suppressing payload logs, model governance, and external validation. Those are unimplemented requirements, not achievements claimed by this project.

**Possible follow-up:** Which operational failures would you prioritize when designing a production readiness plan?

### Q33. What makes the workflow reproducible?

**Short answer:** Deterministic grouping/splitting, seed 42, pinned dependencies, frozen configuration, recorded provenance, and automated checks.

**Deeper answer:** Group-aware CV, train-fitted transformations, and artifact metadata document the chain from data to serving. STEP 21 recorded 410 passing tests, covering the workflow and API. Raw data and the model binary are intentionally Git-ignored. Reproducibility in the recorded environment does not imply byte-identical artifacts across arbitrary platforms or future library versions.

**Possible follow-up:** Why are a fixed random seed and a requirements file helpful but not sufficient by themselves?

### Q34. What is the most important limitation?

**Short answer:** The final internal test is not perfectly analyst-blind because EDA examined the full labeled dataset before the split lock.

**Deeper answer:** I disclose that visibility rather than calling the holdout untouched. The later freeze still prevented post-test model, preprocessing, calibration, and threshold changes, but it does not erase earlier knowledge. There is no external/temporal validation, fairness assessment, bank approval, or production PD claim. Repeated validation use can also introduce selection optimism.

**Possible follow-up:** What additional evidence would support a stronger generalization claim?

## Challenging / Adversarial Questions

### H01. Your AP is only 0.416. Is that good?

It is informative only in context: positives are about 6.6% of the internal test population, and AP measures retrieval quality rather than accuracy. I report the achieved precision/recall trade-offs and development comparisons, but I cannot translate AP into lending usefulness or profitability without operational costs and external evidence.

### H02. Why trust test results if you already saw the full data during EDA?

I would qualify the result as an internal holdout evaluation with prior analyst visibility. Later freeze discipline prevented test-driven changes, but it did not make the analyst retrospectively blind. The limitation is documented, and stronger claims require independent, previously uninspected data.

### H03. Why not remove every duplicate?

Identical predictors can describe different borrowers, and some groups contain conflicting labels. I lacked a reliable identity key or source-system deduplication rule. Keeping rows while isolating feature groups was a defensible evaluation choice, not proof that every duplicate is legitimate.

### H04. Why not deep learning?

I did not evaluate it, so I cannot claim it would perform worse. The implemented scope established linear and tree benchmarks for this tabular dataset. Adding another family would require a new development protocol and resources, not repeated selection on the consumed test.

### H05. Why not optimize F1 directly?

The declared development scenario required at least 50% recall, then maximized precision among eligible thresholds. F1 balances precision and recall but does not guarantee that recall constraint. Neither rule establishes economic optimality without a cost model.

### H06. Why no cost-sensitive objective?

There was no validated financial cost matrix, exposure data, or operational intervention policy. I could demonstrate class weighting and precision-recall trade-offs, but assigning invented financial costs would create unsupported business claims. A business objective would need stakeholder-defined inputs and a new validation plan.

### H07. Why no fairness analysis?

Fairness was outside the implemented evaluation and cannot be inferred from overall performance. Age is present, while attributes such as sex and race are absent; I do not claim the dataset has no potentially sensitive attributes. Appropriate subgroup definitions, permitted use, and fairness criteria would need explicit assessment.

### H08. Why no temporal split?

The modeled schema does not provide a suitable observation timestamp for a defensible temporal split. The two-year target horizon is not an observation date. I report the random group holdout honestly instead of presenting it as evidence of future-time stability.

### H09. Why no external validation?

The project had one labeled source for modeling. Kaggle's unlabeled competition test file cannot supply measured evaluation outcomes, and relabeling an internal partition would not make it external. Independent labeled data would be required for that claim.

### H10. Why so little feature engineering beyond indicators?

I prioritized explicit quality handling and controlled comparisons before adding speculative transformations. The schema does not supply separate credit limits, used balances, or total debt from which to reconstruct arbitrary new financial measures. Further features would need development evidence; the consumed test cannot select them.

### H11. Why use joblib when it can execute code on load?

It preserves the fitted sklearn-compatible pipeline and custom preprocessing state conveniently in this local project. I restrict loading to trusted artifacts, verify bytes against trusted metadata, and document dependency compatibility. That does not make untrusted deserialization safe or solve artifact authenticity by itself.

### H12. Why no Docker?

Containerization was not implemented, and I do not list it as a demonstrated skill here. The current reproduction path uses the documented Python environment and pinned dependencies. A container could support a later delivery workflow but would not replace external model validation or serving security.

### H13. Why no authentication?

The service is bound for local demonstration rather than exposed as a public lending endpoint. Its validated schema and sanitized errors are useful engineering controls, but they are not access control. Authentication and authorization would be required in an appropriate deployment design.

### H14. Why did test metrics improve over validation?

The recorded test sample produced higher ROC-AUC and AP than validation under the frozen model. Sampling variation and differences in case mix are plausible explanations, not demonstrated causes. I did not use that difference to retune, claim a statistically significant gain, or assume deployment will perform better.

### H15. Isn't threshold 0.19 arbitrary?

It is conditional on a documented development scenario, not an arbitrary post-test choice: validation recall of at least 50%, then highest precision. The scenario itself is not a bank-validated utility function. Different legitimate objectives could imply different policies, but this project's frozen thresholds remain unchanged.

## Limitation Questions

**What did you see before the test split was locked?**

The EDA examined aggregate patterns in the full labeled dataset before the final internal split discipline. That gave the analyst information about the eventual test population. I disclose this as prior visibility rather than claiming a perfectly analyst-blind holdout.

**What did the later freeze still accomplish?**

It fixed model, preprocessing, calibration, and threshold decisions before final test evaluation. STEP 15 consumed the test once, and no post-test retuning followed. This protects the stated later evaluation protocol without undoing the earlier EDA exposure.

**What would support a stronger claim?**

Independent, previously uninspected labeled data and a predefined evaluation protocol would provide stronger evidence. Temporal validation would additionally require appropriate timestamps. Neither has been performed, and the existing consumed test should not be repurposed as fresh evidence.

## Role-Specific Interview Emphasis

| Role | Lead With | Technical Depth | What To Avoid Overemphasizing |
| --- | --- | --- | --- |
| Data Analyst | Data quality, EDA findings, visual communication, risk trade-offs | Target definition, missingness, class imbalance, metric interpretation | Detailed estimator settings and serialization internals |
| Data Scientist | Evaluation design and development model comparisons | Group-aware CV, imbalance, threshold/calibration evidence, SHAP limitations | API routing details without a modeling reason |
| ML Engineer | Frozen preprocessing-to-inference contract | Serialization integrity, schema validation, parity, startup lifecycle, reproducibility | Unimplemented cloud infrastructure or production scale |
| Risk Analytics | Serious-delinquency meaning and operating trade-offs | Ranking versus probability quality, thresholds, explainability, governance limits | Treating a benchmark metric as economic or regulatory validation |

## Whiteboard Explanations

### ROC-AUC

Draw the true-positive rate against the false-positive rate as the threshold changes. The area summarizes discrimination and can be interpreted through positive-negative ranking pairs, accounting for ties; it does not report classification accuracy or calibration.

### Average Precision

Draw precision against recall as more observations are selected. AP weights precision by the increments in recall, emphasizing positive retrieval under imbalance; it is not trapezoidal integration and must be interpreted with prevalence.

### Precision

Among observations flagged positive, precision is the fraction that actually have the event. Lower precision implies more false positives among the flags, but does not state how many events were missed.

### Recall

Among all actual event observations, recall is the fraction flagged positive. It measures event detection, while precision describes the reliability of the resulting positive flags.

### F1

F1 is the harmonic mean of precision and recall. It penalizes imbalance between them but does not encode the project's recall constraint or a financial cost matrix.

### Brier Score

For each observation, square the difference between predicted probability and its binary outcome, then average. Lower is better, but the score reflects both discrimination and calibration and needs a population-appropriate comparison.

### Calibration

Group similar predicted probabilities and compare their average predictions with observed event rates. A reliability curve near the diagonal is encouraging on that sample, but it does not guarantee calibration for every subgroup or future population.

### SMOTE

Sketch a minority-class point and its neighbors, then interpolate new points between them. Fit and resample only within training folds; interpolation in count or indicator space may produce unrealistic profiles.

### Group Split

Draw boxes around identical raw feature vectors and move each whole box into one partition. This prevents exact predictor overlap across partitions without claiming the boxes identify people; stratification alone would not impose that constraint.

### SHAP

Start with a reference output and add signed feature contributions to reconstruct a prediction. Here the additive space is the raw margin, with sigmoid applied to the total; these contributions describe model behavior rather than causes.

### Threshold

Draw a cutoff on a score distribution and classify scores at or above it as positive. Moving the cutoff changes precision and recall without refitting the model; this project froze 0.19 and 0.50 before final testing.

## Metrics Cheat Sheet

| Item | Remember |
| --- | --- |
| Dataset / target | Give Me Some Credit; `SeriousDlqin2yrs` |
| Observations / full-data prevalence | 150,000 / 6.684% |
| Train / validation / internal test | 104,998 / 22,486 / 22,516 |
| Frozen model / calibration | XGB-01 / none |
| Validation ROC-AUC / AP | 0.8637 / 0.3813 |
| Final internal-test ROC-AUC / AP | 0.8697 / 0.4161 |
| Threshold 0.19, validation P / R / F1 | 0.3723 / 0.5020 / 0.4275 |
| Threshold 0.19, test P / R / F1 | 0.4034 / 0.5319 / 0.4588 |
| Audit threshold | 0.50 |
| Top global SHAP feature | `RevolvingUtilizationOfUnsecuredLines` |
| Automated tests | 410 passed at STEP 21 |

P/R/F1 are fractions, not percentages. Always name the partition before giving a result. The test is consumed after STEP 15.

## 10 Things to Never Say

| Avoid saying | Replace with |
| --- | --- |
| "The model predicts default with 87% accuracy." | "Internal-test ROC-AUC was 0.8697; it is a ranking metric for the serious-delinquency target, not accuracy." |
| "SHAP proves this feature causes default." | "SHAP attributes the fitted model's prediction to features; it does not establish causality." |
| "0.19 is the optimal bank threshold." | "0.19 was selected on validation under a recall constraint, then frozen for internal evaluation." |
| "This is a regulatory PD model." | "These are internally evaluated serious-delinquency scores without regulatory PD validation." |
| "The test set is still unseen." | "The internal test was consumed in STEP 15; earlier EDA visibility is also disclosed." |
| "96 and 98 are definitely error codes." | "Their meaning is undocumented; the pipeline applies an explicit special-value policy." |
| "SMOTE was bad." | "SMOTE increased recall but had lower precision and F1 than Balanced Logistic in the recorded validation comparison." |
| "The API is production-ready." | "It is a tested local inference service with specified production gaps." |
| "I deployed it to production." | "I implemented and tested a local FastAPI service; production deployment was not performed." |
| "The model is fair." | "Fairness was not assessed; overall metrics and SHAP do not establish it." |

## Mock Interview Round

### M01. Introduce the project

**Interviewer:** What did you build?

**Candidate:** A serious-delinquency scoring portfolio using Give Me Some Credit. I took it from data-quality decisions and model comparisons through frozen evaluation, SHAP explanations, and a tested local API.

### M02. Define the event

**Interviewer:** What does the model actually predict?

**Candidate:** The dataset's two-year serious-delinquency event: 90 days past due or worse. I deliberately avoid calling it bankruptcy or treating the score as an approved lending decision.

### M03. Identify the analytical challenge

**Interviewer:** What made the data challenging?

**Candidate:** Positives were only 6.684%, and there were missing values, undocumented special delinquency values, and repeated predictors. I needed both a quality policy and an evaluation design before comparing models.

### M04. Defend the split

**Interviewer:** Why did you complicate the split with groups?

**Candidate:** The ordinary stratified comparator spread 197 repeated feature groups across partitions. I kept equal raw feature vectors together so that exact predictors did not leak across the split, without pretending they identified the same borrower.

### M05. Explain preprocessing

**Interviewer:** How did you prevent preprocessing leakage?

**Candidate:** Learned medians and scaling were fitted on training or the current fitting fold only. Validation and test received those fitted transformations, and the API later reused the complete frozen pipeline.

### M06. Compare imbalance strategies

**Interviewer:** Did SMOTE solve the imbalance problem?

**Candidate:** It improved Logistic recall, but precision and F1 were lower than Balanced Logistic at the recorded validation threshold. I kept the benchmark and chose the final strategy from broader development evidence rather than assuming resampling was automatically better.

### M07. Explain selection

**Interviewer:** Why choose baseline XGBoost after doing tuning?

**Candidate:** Tuning gave a small AP improvement and slightly lower validation ROC-AUC. The baseline was already a strong candidate, so I preferred that fixed configuration. I would not claim the difference proves one configuration is statistically better.

### M08. Explain the operating point

**Interviewer:** Where did 0.19 come from?

**Candidate:** Validation only: require recall of at least 50%, then maximize precision among eligible thresholds. I froze it before test and kept 0.50 as an audit reference, not as a competing policy selected after testing.

### M09. Separate probability quality

**Interviewer:** Why no calibration layer?

**Candidate:** Sigmoid worsened XGBoost's validation probability metrics, and isotonic's Brier/log-loss gains were very small. I retained the simpler uncalibrated model while documenting that internal probability checks do not establish regulatory PD validity.

### M10. Report the outcome

**Interviewer:** What were the final results?

**Candidate:** On the internal test, ROC-AUC was 0.8697 and AP was 0.4161. At the previously frozen 0.19 threshold, precision was 0.4034 and recall was 0.5319. I did not change decisions based on those results.

### M11. Challenge the holdout

**Interviewer:** But you saw the whole dataset in EDA. Is that a clean test?

**Candidate:** It is not perfectly analyst-blind, and I disclose that. The later freeze prevented post-test tuning, but independent unseen data would be needed for a stronger validation claim.

### M12. Challenge explanations

**Interviewer:** Does high utilization cause the prediction to increase?

**Candidate:** SHAP shows how utilization contributes to this fitted model's output under its explanation setup. I cannot turn that association into a causal intervention claim, especially with correlated features and observational data.

### M13. Explain serving integrity

**Interviewer:** How do you know the API serves the intended model?

**Candidate:** Startup verifies the artifact hash before loading and checks schema and pipeline structure against frozen metadata. Tests also compare API predictions with direct artifact predictions using invented inputs, without accessing real test observations.

### M14. Challenge deployment scope

**Interviewer:** Could a bank use the API tomorrow?

**Candidate:** Not responsibly on this evidence alone. It is a local demo; deployment controls, privacy and governance processes, monitoring, and external validation would need their own work and approval.

### M15. Reflect on the work

**Interviewer:** What would you do next, and what would you leave alone?

**Candidate:** I would seek independent labeled data and clarify business costs and operational requirements before proposing a new development cycle. I would preserve the current frozen results and would not reuse the consumed test to choose improvements.

## Presentation Slide Outline

### Slide 1. Problem & Dataset

- Predict the two-year serious-delinquency target, not bankruptcy.
- Use 150,000 labeled Give Me Some Credit observations.
- Interpret model performance against 6.684% full-data positive prevalence.

**Existing figure:** [Target distribution](figures/target_distribution.png).

**Speaker takeaway:** The project is about meaningful minority-event evaluation, not headline accuracy.

### Slide 2. Data Quality & Leakage

- Document missingness, age zero, special delinquency values, and extreme ratios.
- Keep identical raw feature groups together; exclude source index and target from predictors.
- Learn preprocessing statistics only inside training/fitting folds.
- Disclose that EDA preceded the final split lock.

**Existing figure:** [Missing values](figures/missing_values.png); the group-overlap evidence is a table in [data quality decisions](data_quality_decisions.md).

**Speaker takeaway:** Explicit quality and partition rules make the evaluation easier to defend.

### Slide 3. Modeling Strategy

- Establish Logistic Regression and Random Forest references before boosting comparisons.
- Use shared group-aware training CV for candidate comparison and targeted tuning.
- Evaluate class weighting and SMOTE within fitting data only.

**Existing figure:** [Logistic imbalance validation comparison](figures/logistic_imbalance_validation_metrics.png).

**Speaker takeaway:** Each experiment answers a development question without changing the evaluation boundary.

### Slide 4. Model Comparison

- Compare training CV and validation evidence for XGB-01, XGB-07, and LGBM-02.
- Explain why marginal tuning gains did not justify changing the final candidate.
- Keep XGB-01 frozen; do not claim statistically proven superiority over LightGBM.

**Existing figure:** [Tuned boosting validation comparison](figures/tuned_boosting_validation_comparison.png).

**Speaker takeaway:** Final selection balances ranking evidence and simplicity rather than a single maximum metric.

### Slide 5. Threshold & Calibration

- Select 0.19 on validation under recall >= 50%, then highest precision.
- Keep 0.50 as the audit reference and freeze both before testing.
- Retain no calibration layer after comparing sigmoid and isotonic alternatives.

**Existing figures:** [Threshold trade-off](figures/threshold_precision_recall_tradeoff.png) and [calibration reliability](figures/calibration_reliability_curves.png).

**Speaker takeaway:** Ranking, threshold decisions, and probability quality are separate questions.

### Slide 6. Final Test Results

- Report internal-test ROC-AUC 0.8697 and AP 0.4161.
- At 0.19, report precision 0.4034, recall 0.5319, and F1 0.4588.
- State that test was consumed in STEP 15 and prompted no retuning.

**Existing figure:** [Final internal-test PR curve](figures/final_test_pr_curve.png).

**Speaker takeaway:** These are outcomes of frozen development choices, not inputs to another tuning round.

### Slide 7. Explainability

- Explain validation predictions with global and local SHAP.
- Identify revolving utilization as the top global feature by mean absolute contribution.
- Explain raw-margin additivity and avoid causal or legal adverse-action claims.

**Existing figures:** [Global SHAP bar](figures/shap_global_bar.png) and [local case near 0.50](figures/shap_local_06_near_050.png).

**Speaker takeaway:** Explanations make model behavior inspectable without proving causality.

### Slide 8. API, Limitations & Takeaways

- Serve ten raw features through a frozen, hash-verified pipeline.
- Note 410 passing tests and the distinction between inference correctness and model validity.
- Disclose prior EDA visibility, absent external/fairness validation, and local-only serving scope.
- Preserve frozen results; propose future work only within a new evaluation protocol.

**Existing figure:** None needed; use the request-to-score flow and endpoint contract from the [API report](api_implementation.md).

**Speaker takeaway:** Reproducible engineering and honest limitations are as important as the reported score.

## Final Interview Closing

The main thing I learned is that evaluation discipline matters as much as model choice. With imbalanced credit data, ranking, threshold trade-offs, and calibration answer different questions, so one headline score is not enough. SHAP helped me explain model behavior while keeping causal claims separate. Reproducible splits, frozen decisions, and tested serialization and serving made the workflow inspectable. I also learned to state limitations clearly, especially prior EDA visibility and the difference between a local portfolio API and a validated lending system.

## Detailed Evidence Reference

### Final Internal-Test Metrics

| Metric | Recorded result |
| --- | ---: |
| ROC-AUC | 0.869663 |
| Average Precision | 0.416082 |
| Gini | 0.739325 |
| Brier score | 0.047727 |
| Log loss | 0.173379 |
| ECE | 0.005445 |

| Frozen test threshold | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| 0.19 | 0.403366 | 0.531944 | 0.458817 |
| 0.50 | 0.602592 | 0.187626 | 0.286154 |

### Artifact Identity

Frozen model: **XGBoost Baseline / XGB-01**. Calibration: **none**. Development threshold: **0.19**; audit threshold: **0.50**. Say "SHA-256 verification" in an interview; there is no need to memorize the digest:

```text
e99eb083596beac8b6b53c8ac75d90b028dce7bf9229d759a2967709d9084590
```

### Source Map

- [README](../README.md) and [CV/portfolio descriptions](portfolio_descriptions.md): project scope, counts, delivery and limitations.
- [Data quality decisions](data_quality_decisions.md): duplicate/group counts, comparator overlap, split rationale and raw-data policies; this is the historical STEP 5 decision record.
- [Final model selection](final_model_selection.md) and [frozen specification](final_model_selection.json): training CV/validation evidence, calibration comparison and frozen operating rule.
- [Final internal-test evaluation](final_test_evaluation.md) and [test metrics JSON](final_test_results.json): reported holdout metrics and consumption status.
- [Imbalance results](logistic_imbalance_experiments.json) and [calibration results](calibration_results.json): method-specific development comparisons.
- [Global SHAP](shap_global_explainability.md) and [local SHAP](shap_local_explainability.md): output space, rankings, case-selection policy and explanation limits.
- [Serialization audit](model_serialization.md) and [artifact metadata](final_model_artifact_metadata.json): training-only reconstruction, validation load-back checks, integrity and trust boundaries.
- [API implementation](api_implementation.md): inference contract, input semantics, tests and local-service limitations.

No new evaluation, model change, business-impact estimate, deployment claim, or external/fairness validation is represented by this preparation guide. Hypothetical follow-ups describe questions for future work, not completed project features.
