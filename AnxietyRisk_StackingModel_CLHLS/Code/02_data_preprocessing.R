# 02_data_preprocessing.R: 数据预处理（缺失值插补+分类变量编码+分层拆分）
# 依赖包（需提前安装：install.packages(c("mice", "caret", "dplyr", "tidyr", "stringr"))）
library(mice)        # 缺失值随机森林插补（v3.16.0）
library(caret)       # 数据拆分与预处理（v6.0-94）
library(dplyr)       # 数据操作（v1.1.4）
library(tidyr)       # 数据整理（v1.3.0）
library(stringr)     # 字符串处理（v1.5.1）

# --------------------------
# Step 1: 加载筛选后的数据（来自01_data_load.R的输出）
# 确保processed_data文件夹已存在，且包含筛选后的数据集
wave5_eligible <- read.csv("processed_data/wave5_eligible.csv", stringsAsFactors = FALSE)
wave4_eligible <- read.csv("processed_data/wave4_eligible.csv", stringsAsFactors = FALSE)

# 查看数据基本信息（验证数据加载）
cat("=== 数据加载验证 ===\n")
cat(paste("Wave5内部队列样本量:", nrow(wave5_eligible), "\n"))
cat(paste("Wave4外部验证队列样本量:", nrow(wave4_eligible), "\n"))
cat(paste("Wave5焦虑患病率（GAD-7≥5）:", round(mean(wave5_eligible$GAD_7 >= 5) * 100, 1), "%\n"))
cat(paste("Wave4焦虑患病率（GAD-7≥5）:", round(mean(wave4_eligible$GAD_7 >= 5) * 100, 1), "%\n"))

# --------------------------
# Step 2: 缺失值处理（随机森林插补，符合论文2.4节方法）
# 定义插补函数（统一处理两个队列）
impute_missing <- function(data, cohort_name) {
  # 计算插补前缺失率
  pre_missing_rate <- sum(is.na(data)) / (nrow(data) * ncol(data))
  
  # 随机森林插补（m=5个插补数据集，最终取均值）
  set.seed(123)  # 固定随机种子，保证可复现
  imputed_data <- mice(
    data = data,
    method = "rf",  # 随机森林插补
    m = 5,          # 生成5个插补数据集
    maxit = 50,     # 最大迭代次数
    printFlag = FALSE  # 隐藏中间输出
  )
  
  # 提取插补后的数据（合并5个数据集的结果）
  completed_data <- complete(imputed_data, action = "long") %>%
    group_by(.id) %>%
    summarise(across(everything(), mean, na.rm = TRUE)) %>%
    select(-.id)
  
  # 还原分类变量的整数编码（插补后可能为小数，需四舍五入）
  categorical_vars <- c(
    # 生物学维度（序数/二分类变量）
    "age_level", "self_rated_health", "chronic_disease_total", "body_pain_level",
    "eyesight_problem", "hearing_problem", "blood_pressure_sugar", "lost_teeth",
    "BADL_disability", "IADL_disability", "sleep_duration", "physical_activities",
    "falling_down", "fractured_hip", "alcohol_intake", "cigarettes_consumption",
    # 心理维度（序数/二分类变量）
    "CESD_10", "MMSE_status", "TICS_status", "WR_status", "RF_status", "CSI_D_status",
    "health_satisfaction", "air_quality_satisfaction", "children_satisfaction",
    "marriage_satisfaction", "religious_beliefs",
    # 社会维度（序数/二分类/名义变量）
    "address_type", "government_pension", "marital_status", "educational_level",
    "residential_district", "building_type", "steps_to_entrance", "nation",
    "internet_entertainment", "agricultural_work", "fringe_benefits", "pension_type",
    "household_debts", "household_type", "house_elevator",
    # 结局变量
    "GAD_7"
  )
  
  completed_data <- completed_data %>%
    mutate(across(all_of(categorical_vars), round, digits = 0)) %>%
    mutate(across(all_of(categorical_vars), as.integer))
  
  # 计算插补后缺失率
  post_missing_rate <- sum(is.na(completed_data)) / (nrow(completed_data) * ncol(completed_data))
  
  # 打印插补结果
  cat(paste("\n===", cohort_name, "缺失值插补结果 ===\n"))
  cat(paste("插补前整体缺失率:", round(pre_missing_rate * 100, 2), "%\n"))
  cat(paste("插补后整体缺失率:", round(post_missing_rate * 100, 2), "%\n"))
  
  return(completed_data)
}

# 对两个队列分别进行缺失值插补
wave5_imputed <- impute_missing(wave5_eligible, "Wave5内部队列")
wave4_imputed <- impute_missing(wave4_eligible, "Wave4外部队列")

# --------------------------
# Step 3: 分类变量编码标准化（符合论文2.4节编码规则）
# 编码规则：
# - 序数变量：保持原有顺序编码（preserve order）
# - 名义变量：转为虚拟变量（dummy coding），指定参考组（ref）
encode_categorical <- function(data) {
  # 1. 定义变量类型（基于Table S1）
  ordinal_vars <- c(
    "age_level", "self_rated_health", "chronic_disease_total", "body_pain_level",
    "eyesight_problem", "hearing_problem", "sleep_duration", "CESD_10",
    "MMSE_status", "TICS_status", "WR_status", "RF_status", "CSI_D_status",
    "children_satisfaction", "marriage_satisfaction", "educational_level",
    "steps_to_entrance", "household_debts", "house_elevator"
  )
  
  nominal_vars <- list(
    address_type = list(ref = 1),          # 参考组：1=家庭住房
    marital_status = list(ref = 1),        # 参考组：1=已婚且配偶在场
    residential_district = list(ref = 1),   # 参考组：1=城镇中心
    building_type = list(ref = 1),         # 参考组：1=单元楼/独栋住宅
    household_type = list(ref = 1)         # 参考组：1=单元楼
  )
  
  binary_vars <- c(
    "blood_pressure_sugar", "lost_teeth", "BADL_disability", "IADL_disability",
    "physical_activities", "falling_down", "fractured_hip", "alcohol_intake",
    "cigarettes_consumption", "health_satisfaction", "air_quality_satisfaction",
    "religious_beliefs", "government_pension", "nation", "internet_entertainment",
    "agricultural_work", "fringe_benefits", "pension_type"
  )
  
  # 2. 序数变量：确保为因子类型（保留顺序）
  data <- data %>%
    mutate(across(all_of(ordinal_vars), as.factor)) %>%
    mutate(across(all_of(ordinal_vars), ~factor(., ordered = TRUE)))
  
  # 3. 二分类变量：转为0/1编码（原编码1=否，2=是）
  data <- data %>%
    mutate(across(all_of(binary_vars), ~ifelse(. == 1, 0, 1)))
  
  # 4. 名义变量：生成虚拟变量（排除参考组，避免多重共线性）
  for (var in names(nominal_vars)) {
    ref_val <- nominal_vars[[var]]$ref
    # 生成虚拟变量
    dummy_vars <- model.matrix(~ . - 1, data = data[, var, drop = FALSE])
    # 重命名虚拟变量
    colnames(dummy_vars) <- str_replace(colnames(dummy_vars), var, paste0(var, "_cat"))
    # 排除参考组对应的列
    dummy_vars <- dummy_vars[, !colnames(dummy_vars) == paste0(var, "_cat", ref_val), drop = FALSE]
    # 合并到原数据
    data <- cbind(data, dummy_vars)
    # 删除原始名义变量
    data <- data[, !names(data) == var, drop = FALSE]
  }
  
  # 5. 结局变量：转为二分类（0=无焦虑，1=有焦虑；GAD-7≥5为阳性）
  data <- data %>%
    mutate(anxiety = ifelse(GAD_7 >= 5, 1, 0)) %>%
    select(-GAD_7)  # 删除原始GAD-7分数，保留二分类结局
  
  return(data)
}

# 对插补后的数据进行编码
wave5_encoded <- encode_categorical(wave5_imputed)
wave4_encoded <- encode_categorical(wave4_imputed)

# 验证编码结果
cat("\n=== 分类变量编码验证 ===\n")
cat(paste("Wave5编码后变量数:", ncol(wave5_encoded), "\n"))
cat(paste("Wave4编码后变量数:", ncol(wave4_encoded), "\n"))
cat(paste("Wave5编码后焦虑阳性数:", sum(wave5_encoded$anxiety), "\n"))
cat(paste("Wave4编码后焦虑阳性数:", sum(wave4_encoded$anxiety), "\n"))

# --------------------------
# Step 4: 10次分层随机拆分（保持焦虑患病率16.5%，符合论文2.4节）
set.seed(456)  # 固定随机种子，保证可复现
n_splits <- 10  # 10次拆分
split_list <- list()  # 存储每次拆分的索引

# 创建输出文件夹（存储拆分后的数据）
if (!dir.exists("processed_data/splits")) {
  dir.create("processed_data/splits", recursive = TRUE)
}

# 分层拆分函数（按焦虑状态分层）
split_data <- function(encoded_data, n_splits, cohort_name) {
  for (i in 1:n_splits) {
    # 分层拆分（70%训练集，30%测试集）
    train_index <- createDataPartition(
      y = encoded_data$anxiety,
      p = 0.7,
      list = FALSE,
      times = 1
    )
    train_data <- encoded_data[train_index, ]
    test_data <- encoded_data[-train_index, ]
    
    # 保存本次拆分的数据
    write.csv(
      train_data,
      paste0("processed_data/splits/", cohort_name, "_train_split", i, ".csv"),
      row.names = FALSE
    )
    write.csv(
      test_data,
      paste0("processed_data/splits/", cohort_name, "_test_split", i, ".csv"),
      row.names = FALSE
    )
    
    # 保存拆分索引
    split_list[[i]] <- list(
      split_id = i,
      train_n = nrow(train_data),
      test_n = nrow(test_data),
      train_anxiety_rate = round(mean(train_data$anxiety) * 100, 1),
      test_anxiety_rate = round(mean(test_data$anxiety) * 100, 1)
    )
    
    # 打印本次拆分信息
    cat(paste("\n===", cohort_name, "第", i, "次拆分 ===\n"))
    cat(paste("训练集样本量:", nrow(train_data), "，焦虑患病率:", round(mean(train_data$anxiety) * 100, 1), "%\n"))
    cat(paste("测试集样本量:", nrow(test_data), "，焦虑患病率:", round(mean(test_data$anxiety) * 100, 1), "%\n"))
  }
  
  # 转换拆分信息为数据框并保存
  split_summary <- do.call(rbind, lapply(split_list, function(x) data.frame(x)))
  write.csv(
    split_summary,
    paste0("processed_data/splits/", cohort_name, "_split_summary.csv"),
    row.names = FALSE
  )
  
  return(split_summary)
}

# 对内部队列（Wave5）进行10次拆分（外部队列Wave4无需拆分，直接用于验证）
wave5_split_summary <- split_data(wave5_encoded, n_splits, "wave5")

# 保存外部队列编码后的数据（用于后续外部验证）
write.csv(
  wave4_encoded,
  "processed_data/wave4_encoded_validation.csv",
  row.names = FALSE
)

# --------------------------
# Step 5: 保存预处理后的核心数据（供后续脚本调用）
# 1. 插补+编码后的完整数据
write.csv(wave5_encoded, "processed_data/wave5_imputed_encoded.csv", row.names = FALSE)
write.csv(wave4_encoded, "processed_data/wave4_imputed_encoded.csv", row.names = FALSE)

# 2. 10次拆分的摘要信息
write.csv(wave5_split_summary, "processed_data/wave5_split_summary.csv", row.names = FALSE)

# --------------------------
# 最终验证信息打印
cat("\n=== 数据预处理完成 ===\n")
cat("输出文件说明：\n")
cat("1. processed_data/wave5_imputed_encoded.csv: Wave5插补+编码后完整数据\n")
cat("2. processed_data/wave4_imputed_encoded.csv: Wave4插补+编码后完整数据\n")
cat("3. processed_data/splits/: 10次分层拆分的训练集/测试集\n")
cat("4. processed_data/wave5_split_summary.csv: 10次拆分的样本量和患病率摘要\n")
cat("5. processed_data/wave4_encoded_validation.csv: Wave4外部验证数据集\n")
cat("\n下一步：运行03_feature_selection.R进行特征选择\n")