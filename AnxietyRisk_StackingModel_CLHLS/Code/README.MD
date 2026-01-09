ANXIETYRISK_STACKINGMODEL_CLHLS
This repository contains complete R code, supplementary materials, and documentation for the study: "Development of an Interpretable Stacking Model for Anxiety Risk Prediction in Elderly Women Receiving Home-Based Care Using CLHLS Data"
1. PROJECT OVERVIEW
Core Objective
Develop an interpretable concurrent risk assessment model to predict anxiety symptoms (GAD-7 ≥ 5) in relatively independent, rural-dominant elderly women (≥70 years) receiving home-based care in eastern/central China.
Key Features
Theoretical Framework: Based on the Bio-Psycho-Social model (40 candidate variables across biological, psychological, and social dimensions).
Methodology: Integrates data balancing (SMOTE, BorderlineSMOTE, SMOTE-ENN+Tomek), Elastic Net/Group LASSO feature selection, and a Stacking ensemble model (5 base algorithms: LR, kNN, RF, SVM, NN).
Interpretability: SHAP/LIME analysis + ICE curves to explain feature importance and model logic.
Validation: Internal (10-fold stratified splits) + external (CLHLS Wave 4) validation to ensure robustness.
Main Result
Core Predictors: Body pain level, sleep duration, self-rated health, hearing problem, air quality satisfaction.
2. ENVIRONMENT SETUP
Required Software
R version: 4.3.0 (strictly recommended to avoid compatibility issues)
RStudio (optional but recommended for code execution)
Required R Packages & Versions
Install the following packages with specified versions before running the code:
install.packages(c(
"mice" = "3.16.0",
"glmnet" = "4.1-8",
"grpreg" = "3.4.1",
"randomForest" = "4.7-1.1",
"e1071" = "1.7-13",
"nnet" = "7.3-19",
"caretEnsemble" = "2.0.1",
"pROC" = "1.18.4",
"caret" = "6.0-94",
"shapviz" = "0.10.0",
"lime" = "0.5.3",
"simstudy" = "0.7.8",
"ggplot2" = "3.4.4",
"dplyr" = "1.1.4",
"tidyr" = "1.3.0",
"gridExtra" = "2.3",
"corrplot" = "0.92",
"broom" = "1.0.5",
"reshape2" = "1.4.4"
))
3. FILE STRUCTURE & DESCRIPTION
All files follow a sequential workflow (run in numerical order). Below is the full file structure and core functions:
File Path: Code/01_data_load.R
Core Function: Load CLHLS Wave4/Wave5 data, screen eligible participants.
Corresponding Study Section: 2.1 Data Sources
File Path: Code/02_data_preprocessing.R
Core Function: Handle missing values (random forest imputation), standardize categorical coding, and split data into 10 stratified train-test sets.
Corresponding Study Section: 2.4 Data Preprocessing
File Path: Code/03_feature_selection.R
Core Function: Select key variables via Elastic Net-CV and validate with Group LASSO .
Corresponding Study Section: 2.4 Feature Selection
File Path: Code/04_data_balancing.R
Core Function: Apply 3 data-balancing techniques to address class imbalance.
Corresponding Study Section: 2.4 Data Balancing
File Path: Code/05_model_training.R
Core Function: Train 5 base models and Stacking ensemble .
Corresponding Study Section: 2.5 Model Construction
File Path: Code/06_model_evaluation.R
Core Function: Calculate performance metrics and plot ROC/calibration/precision-recall curves.
Corresponding Study Section: 2.6 Model Evaluation
File Path: Code/07_model_interpretation.R
Core Function: Conduct SHAP/LIME analysis and plot ICE curves to explain feature importance and individual heterogeneity.
Corresponding Study Section: 2.6 Model Interpretation
File Path: Code/08_simulation_sensitivity.R
Core Function: Validate variable selection accuracy and assess survivorship bias.
Corresponding Study Section: 2.8 Simulation Study
File Path: Code/09_result_reproduction.R
Core Function: Reproduce core tables  and figures from the study.
Corresponding Study Section: 3. Results
File Path: Docs/Model_Parameters.csv
Core Function: Optimal hyperparameters for all models .
File Path: Docs/Feature_Dictionary.csv
Core Function: Definitions and coding rules for 18 selected key variables.
File Path: Supplementary/Supplementary_Tables.xlsx
Core Function: All supplementary tables from the study.
Corresponding Study Section: Supplementary Materials
File Path: processed_data/
Core Function: Auto-generated folder for intermediate/output data.
File Path: LICENSE
Core Function: MIT License.
4. STEP-BY-STEP USAGE GUIDE
Prerequisite: Prepare CLHLS Data
Apply for CLHLS Wave4 (2005) and Wave5 (2008) data from the CLHLS official website (http://clhls.pku.edu.cn/en/).
Extract the core variables and save as two CSV files: CLHLS_Wave4_2005.csv and CLHLS_Wave5_2008.csv.
Ensure the files include the following columns : ID, age_level, gender, address_type, GAD_7, and all 40 candidate variables.
Run the Code (Sequential Order)
Set Working Directory
Open R/RStudio, set the working directory to the root of this repository:
setwd("path/to/AnxietyRisk_StackingModel_CLHLS")
Step 1: Data Loading & Screening
Run Code/01_data_load.R to:
Load CLHLS data.
Screen eligible participants.
Output: processed_data/wave5_eligible.csv and processed_data/wave4_eligible.csv.
Step 2: Data Preprocessing
Run Code/02_data_preprocessing.R to:
Impute missing values (random forest method).
Standardize categorical coding (ordinal: sequential; nominal: dummy).
Split Wave5 data into 10 stratified train-test sets (preserve 16.5% anxiety prevalence).
Output: processed_data/wave5_imputed_encoded.csv and processed_data/splits/ (10 splits).
Step 3: Feature Selection
Run Code/03_feature_selection.R to:
Select 18 key variables via Elastic Net-CV.
Validate with Group LASSO (sensitivity analysis).
Output: processed_data/feature_selection/final_selected_features.csv.
Step 4: Data Balancing
Run Code/04_data_balancing.R to:
Apply 3 balancing techniques to the 10 train sets.
Output: processed_data/balanced_data/ (separate folders for SMOTE/BorderlineSMOTE/SET).
Step 5: Model Training
Run Code/05_model_training.R to:
Train 5 base models and Stacking ensemble.
Output: processed_data/model_training/ (saved models + prediction results).
Step 6: Model Evaluation
Run Code/06_model_evaluation.R to:
Calculate internal/external validation metrics.
Plot ROC/calibration/precision-recall curves.
Output: processed_data/model_results/ (metrics) and processed_data/model_plots/ (curves).
Step 7: Model Interpretation
Run Code/07_model_interpretation.R to:
Generate SHAP summary plots, LIME analysis, and ICE curves.
Output: processed_data/model_interpretation/ (plots + SHAP values).
Step 8: Simulation & Sensitivity Analysis
Run Code/08_simulation_sensitivity.R to:
Validate variable selection accuracy.
Assess survivorship bias (simulate deceased group’s anxiety status).
Output: processed_data/simulation_sensitivity/ (results).
Step 9: Reproduce Study Results
Run Code/09_result_reproduction.R to:
Reproduce core tables and figures.
Output: processed_data/final_results/ (tables + combined figures).
Key Notes
All random seeds are fixed for reproducibility.
The default data-balancing method is SET (SMOTE-ENN+Tomek), which achieved the best performance (AUC=0.862).
To use other balancing methods (SMOTE/BorderlineSMOTE), modify balance_method in Code/05_model_training.R.
5.  DATA AVAILABILITY
CLHLS Raw Data: Available from the CLHLS Center (http://clhls.pku.edu.cn/en/) upon reasonable request and with CLHLS permission (we cannot share raw data directly due to license restrictions).
Code & Supplementary Materials: Publicly accessible in this repository (all scripts, tables, and documentation).
Processed Data: Generated automatically when running the code (stored in processed_data/).
6. AUTHORS & CONTACT
Corresponding Author: Kai Wang (wangkai0552@126.com)
Affiliation: School of Health Management, Bengbu Medical University, China
For code-related issues (e.g., bugs, compatibility), please open an issue in this repository or contact the corresponding author.
7. CITATION
If you use this code or data in your research, please cite the study as follows:
Wang K, Du Y. Development of an Interpretable Stacking Model for Anxiety Risk Prediction in Elderly Women Receiving Home-Based Care Using CLHLS Data.
8. LICENSE
This project is licensed under the MIT License - see the LICENSE file for details (open for academic use; commercial use requires permission from the authors).
ACKNOWLEDGEMENTS
We thank the Center for China Longitudinal Health Longevity Survey (CLHLS) for providing the dataset.