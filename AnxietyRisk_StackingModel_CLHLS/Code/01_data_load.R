# 01_data_load.R: Load CLHLS data and screen eligible participants
# Dependent packages (install first if not available: install.packages(c("readr", "dplyr", "tidyr")))
library(readr)
library(dplyr)
library(tidyr)

# --------------------------
# Step 1: Set file paths (replace with your local file paths or data access links)
# Note: CLHLS wave4 (2005) and wave5 (2008) data should be in CSV format with consistent variable names
wave5_path <- "CLHLS_Wave5_2008.csv"  # Internal cohort (model training/validation)
wave4_path <- "CLHLS_Wave4_2005.csv"  # External validation cohort

# --------------------------
# Step 2: Load data (handle possible missing column warnings if any)
# Wave5 data (internal cohort: model development & internal validation)
wave5_data <- read_csv(wave5_path, show_col_types = FALSE) %>%
  # Keep only variables required for screening (consistent with Table S1)
  select(
    ID,  # Unique participant ID
    age_level,  # Age group (1=60-69, 2=70-79, 3=≥80)
    gender,  # Gender (1=Male, 2=Female)
    address_type,  # Housing type (1=Family housing, 2=Nursing home, 3=Hospital, 4=Other)
    GAD_7,  # GAD-7 score (outcome variable)
    # Include all 40 candidate variables (consistent with Table S1)
    self_rated_health, chronic_disease_total, body_pain_level, eyesight_problem, hearing_problem,
    blood_pressure_sugar, lost_teeth, BADL_disability, IADL_disability, sleep_duration,
    physical_activities, falling_down, fractured_hip, alcohol_intake, cigarettes_consumption,
    CESD_10, MMSE_status, TICS_status, WR_status, RF_status, CSI_D_status,
    health_satisfaction, air_quality_satisfaction, children_satisfaction, marriage_satisfaction,
    religious_beliefs, government_pension, marital_status, educational_level, residential_district,
    building_type, steps_to_entrance, nation, internet_entertainment, agricultural_work,
    fringe_benefits, pension_type, household_debts, household_type, house_elevator
  )

# Wave4 data (external validation cohort)
wave4_data <- read_csv(wave4_path, show_col_types = FALSE) %>%
  select(
    ID, age_level, gender, address_type, GAD_7,
    self_rated_health, chronic_disease_total, body_pain_level, eyesight_problem, hearing_problem,
    blood_pressure_sugar, lost_teeth, BADL_disability, IADL_disability, sleep_duration,
    physical_activities, falling_down, fractured_hip, alcohol_intake, cigarettes_consumption,
    CESD_10, MMSE_status, TICS_status, WR_status, RF_status, CSI_D_status,
    health_satisfaction, air_quality_satisfaction, children_satisfaction, marriage_satisfaction,
    religious_beliefs, government_pension, marital_status, educational_level, residential_district,
    building_type, steps_to_entrance, nation, internet_entertainment, agricultural_work,
    fringe_benefits, pension_type, household_debts, household_type, house_elevator
  )

# --------------------------
# Step 3: Screen eligible participants (consistent with Section 2.1 of the manuscript)
# Eligibility criteria:
# 1. Aged ≥70 years (age_level: 2=70-79, 3=≥80)
# 2. Female (gender=2)
# 3. Receiving home-based care (address_type=1: Family housing)
# 4. No missing GAD-7 score (outcome variable)
# 5. Missing individual variables ≤20% (per participant)

# Define screening function
screen_participants <- function(data) {
  data %>%
    # Criterion 1: Aged ≥70 years
    filter(age_level %in% c(2, 3)) %>%
    # Criterion 2: Female
    filter(gender == 2) %>%
    # Criterion 3: Home-based care (family housing)
    filter(address_type == 1) %>%
    # Criterion 4: No missing GAD-7 score
    filter(!is.na(GAD_7)) %>%
    # Criterion 5: Missing variables ≤20% per participant
    rowwise() %>%
    mutate(missing_rate = sum(is.na(c_across(everything()))) / ncol(.)) %>%
    ungroup() %>%
    filter(missing_rate ≤ 0.2) %>%
    # Remove temporary missing_rate column
    select(-missing_rate)
}

# Apply screening to both cohorts
wave5_eligible <- screen_participants(wave5_data)
wave4_eligible <- screen_participants(wave4_data)

# --------------------------
# Step 4: Save screened data for subsequent analysis
# Create output directory (if not exists)
if (!dir.exists("processed_data")) {
  dir.create("processed_data")
}

# Save eligible data as CSV (ready for 02_data_preprocessing.R)
write_csv(wave5_eligible, "processed_data/wave5_eligible.csv")
write_csv(wave4_eligible, "processed_data/wave4_eligible.csv")

# Print summary for verification
cat("Screening Summary:\n")
cat(paste("Wave5 (Internal Cohort):", nrow(wave5_eligible), "eligible participants\n"))
cat(paste("Wave4 (External Cohort):", nrow(wave4_eligible), "eligible participants\n"))
cat(paste("Anxiety prevalence (GAD-7≥5) in Wave5:", round(mean(wave5_eligible$GAD_7 ≥ 5) * 100, 1), "%\n"))
cat(paste("Anxiety prevalence (GAD-7≥5) in Wave4:", round(mean(wave4_eligible$GAD_7 ≥ 5) * 100, 1), "%\n"))

# --------------------------
# Notes:
# 1. Replace wave5_path and wave4_path with your actual data file paths (local or server links)
# 2. Ensure variable names match CLHLS data (adjust if your data uses different naming conventions)
# 3. The code automatically creates a "processed_data" folder to store screened data
# 4. Dependent packages: readr (v2.1.4), dplyr (v1.1.4), tidyr (v1.3.0)