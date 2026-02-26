# Medical Transcription Specialty Classification
[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20Series%20S6E2-20BEFF?logo=kaggle&logoColor=white)](https://kaggle.com/competitions/playground-series-s6e2)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview
Medical data is strictly protected under HIPAA privacy regulations. This limits data availability for machine learning research. This project bypasses this limitation. It utilizes publicly available medical transcription samples scraped from `mtsamples.com`. The primary objective is to accurately classify medical specialties. It uses only the raw clinical transcription text.

## Dataset
The dataset consists of clinical notes categorized by medical specialty.
* **Size:** 2,377 unique sample names and 2,358 unique transcriptions.
* **Data Quality:** The keywords feature contains 21% null values and 2% empty values.
* **Distribution:** The target variable is heavily skewed. Surgery constitutes 22% of the dataset. Consultations account for 10%. The remaining 68% falls into a long tail of other specialties.

### Medical Specialty Frequencies (Excerpt)
| Specialty | Count |
| :--- | :--- |
| Radiology | 218 |
| General Medicine | 207 |
| Gastroenterology | 179 |
| Neurology | 178 |
| SOAP / Chart / Progress Notes | 133 |
| Urology | 125 |

## Methodology
The pipeline processes the text and evaluates multiple classification frameworks.
* **Data Splitting:** Data is partitioned into training, validation, and test sets.
* **Classical Baselines:** We evaluate logistic regression, Support Vector Machine (SVM), random forest, and gradient boosting. We use 5-fold cross-validation.
* **Transformer Baseline:** We fine-tune DistilBERT for sequence classification. This captures deeper semantic context.

## Results
Macro-F1 is prioritized as the primary diagnostic metric. High overall accuracy is largely driven by frequent classes like Surgery. Macro-F1 provides a balanced evaluation across the long-tail label distribution.

### Model Performance Leaderboard (5-Fold CV Means)
| Model | Split | Accuracy | Macro-F1 | Source |
| :--- | :--- | :--- | :--- | :--- |
| **svm** | cv5_mean | 0.154332 | 0.120294 | classical_cv |
| **rf** | cv5_mean | 0.146776 | 0.065472 | classical_cv |
| **logreg** | cv5_mean | 0.274422 | 0.063190 | classical_cv |
| **gb** | cv5_mean | 0.151032 | 0.040418 | classical_cv |

![Cross Validation Accuracy by Fold Across Models](image.png)

Logistic regression maintains the highest accuracy across all five folds. The range is 0.25 to 0.29. Gradient boosting accuracy drops sharply at fold five. SVM and random forest maintain stable accuracy trajectories near 0.15.

![Top 20 Significant Predictors (Magnitude): svm](image-1.png)

The SVM model relies heavily on specific clinical tokens to form decision boundaries. The top five predictive tokens by absolute magnitude are "sleep", "eye", "operation", "discharge", and "thyroid".

## Conclusion
Logistic regression achieves the highest overall cross-validation accuracy. SVM achieves the highest macro-F1 score. The severe class imbalance dictates these outcomes. Models struggle to generalize across the rare specialties.

## Limitations and Future Work
* **Class Imbalance:** Extreme label skew limits macro-F1 performance across all classical baselines.
* **Metric Consolidation:** Held-out test metrics are currently unconsolidated. They appear as `NaN` in the broader pipeline outputs.
* **Transformer Integration:** DistilBERT training requires patches for Transformers v5 API changes. Final metrics cannot be recorded until this is fixed.
