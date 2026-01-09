markdown
# Anxiety Risk Prediction Stacking Model (CLHLS Data)
This repository contains R code, supplementary tables, and figures for the paper:  
**Development of an Interpretable Stacking Model for Anxiety Risk Prediction in Elderly Women Receiving Home-Based Care Using CLHLS Data**  
Corresponding author: Kai Wang (wangkai0552@126.com)

## 1. Project Overview
This study develops an interpretable Stacking model to assess concurrent anxiety risk in rural-dominant, home-based elderly women (≥70 years) in eastern/central China, using CLHLS wave4 (2005) and wave5 (2008) data.

## 2. Environment Setup
To run the code, install these tools first:
- R version: 4.3.0 (must use this version to avoid compatibility issues)
- Required R packages (install via `install.packages(c("包名1", "包名2"))`):
  - mice(3.16.0), glmnet(4.1-8), grpreg(3.4.1)
  - DMwR(0.4.1), caretEnsemble(2.0.1), pROC(1.18.4)
  - ggplot2(3.4.4), lime(0.5.3), shapviz(0.10.0), simstudy(0.7.8)

## 3. How to Run the Code (Step-by-Step)
| Step | File Path       | Purpose                                  |
|------|-----------------|------------------------------------------|
| 1    | Code/01_data_load.R | Load and screen CLHLS data               |
| 2    | Code/02_data_preprocessing.R | Impute missing values + split data       |
| 3    | Code/03_feature_selection.R | Select key variables via Elastic Net     |
| 4    | Code/04_data_balancing.R | Balance training data (SMOTE-ENN+Tomek)  |
| 5    | Code/05_model_training.R | Train base models + Stacking ensemble    |
| 6    | Code/06_model_evaluation.R | Calculate performance metrics (AUC/F1)   |
| 7    | Code/07_model_interpretation.R | SHAP/LIME analysis for model interpretability |
| 8    | Code/08_simulation_sensitivity.R | Validate variable selection and survivorship bias |
| 9    | Code/09_result_reproduction.R | Reproduce paper figures/metrics          |

## 4. Data Availability
- **Code & Supplementary Materials**: Publicly accessible in this GitHub repository (https://github.com/KaiWang-BMU/AnxietyRisk_StackingModel_CLHLS).  
- **Raw CLHLS Data**: Available from the CLHLS center (http://clhls.pku.edu.cn/en/) upon request and with CLHLS permission (we cannot share raw data directly due to license restrictions).

## 5. Contact
For code issues, email Kai Wang (wangkai0552@126.com)
