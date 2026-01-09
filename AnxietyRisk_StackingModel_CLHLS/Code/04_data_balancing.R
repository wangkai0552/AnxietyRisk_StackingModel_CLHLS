# 04_data_balancing.R: 数据平衡处理（SMOTE + BorderlineSMOTE + SMOTE-ENN+Tomek）
# 依赖包（需提前安装：install.packages(c("DMwR", "unbalanced", "dplyr", "tidyr"))）
library(DMwR)         # SMOTE实现（v0.4.1）
library(unbalanced)   # BorderlineSMOTE、SMOTE-ENN+Tomek实现（v2.0）
library(dplyr)        # 数据操作（v1.1.4）
library(tidyr)        # 数据整理（v1.3.0）

# --------------------------
# Step 1: 加载数据（来自02_data_preprocessing.R和03_feature_selection.R的输出）
# 1. 加载10次拆分的训练集
split_dir <- "processed_data/splits"
if (!dir.exists(split_dir)) {
  stop("请先运行02_data_preprocessing.R生成拆分数据！")
}

# 2. 加载最终选中的特征列表
feature_file <- "processed_data/feature_selection/final_selected_features.csv"
if (!file.exists(feature_file)) {
  stop("请先运行03_feature_selection.R生成选中特征列表！")
}
selected_features <- read.csv(feature_file)$selected_variable
outcome_var <- "anxiety"  # 结局变量（0=无焦虑，1=有焦虑）

# 3. 加载外部验证数据集（无需平衡，仅用于后续验证）
wave4_encoded <- read.csv("processed_data/wave4_encoded_validation.csv", stringsAsFactors = FALSE)

# --------------------------
# Step 2: 定义三种数据平衡方法（符合论文2.4节）
# 方法1: SMOTE（合成少数类过采样）
smote_balance <- function(train_data, selected_features, outcome_var) {
  # 构建特征矩阵和结局向量
  x <- train_data[, selected_features, drop = FALSE]
  y <- train_data[[outcome_var]]
  
  # SMOTE过采样（设置比例使少数类占比约40%，平衡效果更优）
  set.seed(2024)  # 固定随机种子，保证可复现
  smote_data <- SMOTE(
    x = x,
    y = as.factor(y),
    perc.over = 200,    # 少数类过采样比例（原始少数类×3）
    perc.under = 150    # 多数类欠采样比例（原始多数类×1.5）
  )
  
  # 整理输出格式（结局变量转为数值型）
  balanced_data <- smote_data %>%
    mutate(!!outcome_var := as.integer(as.character(class))) %>%
    select(-class) %>%
    relocate(!!outcome_var, .after = last_col())  # 结局变量移至最后一列
  
  return(balanced_data)
}

# 方法2: BorderlineSMOTE（边界少数类过采样）
borderline_smote_balance <- function(train_data, selected_features, outcome_var) {
  x <- train_data[, selected_features, drop = FALSE]
  y <- train_data[[outcome_var]]
  
  # 转换为unbalanced包要求的格式
  data_mat <- cbind(x, y = y)
  
  set.seed(2024)
  blsmote_data <- ubBalance(
    data = data_mat,
    type = "borderlineSMOTE",  # 方法类型：BorderlineSMOTE
    positive = 1,              # 少数类标签（焦虑=1）
    percOver = 200,            # 少数类过采样比例
    percUnder = 150,           # 多数类欠采样比例
    k = 5                      # 近邻数（默认5，符合论文参数）
  )$data
  
  # 整理输出格式
  balanced_data <- blsmote_data %>%
    rename(!!outcome_var := y) %>%
    mutate(!!outcome_var := as.integer(!!outcome_var))
  
  return(balanced_data)
}

# 方法3: SMOTE-ENN+Tomek（SET，过采样+噪声移除+冗余样本删除）
set_balance <- function(train_data, selected_features, outcome_var) {
  x <- train_data[, selected_features, drop = FALSE]
  y <- train_data[[outcome_var]]
  data_mat <- cbind(x, y = y)
  
  set.seed(2024)
  # 第一步：SMOTE过采样
  smote_step <- ubBalance(
    data = data_mat,
    type = "SMOTE",
    positive = 1,
    percOver = 200,
    percUnder = 150,
    k = 5
  )$data
  
  # 第二步：ENN（Edited Nearest Neighbors）移除噪声样本
  enn_step <- ubBalance(
    data = smote_step,
    type = "ENN",
    positive = 1,
    k = 3  # 近邻数=3（论文推荐）
  )$data
  
  # 第三步：Tomek Links移除冗余样本
  set_data <- ubBalance(
    data = enn_step,
    type = "Tomek",
    positive = 1
  )$data
  
  # 整理输出格式
  balanced_data <- set_data %>%
    rename(!!outcome_var := y) %>%
    mutate(!!outcome_var := as.integer(!!outcome_var))
  
  return(balanced_data)
}

# --------------------------
# Step 3: 对10次拆分的训练集分别应用三种平衡方法
n_splits <- 10
balance_methods <- list(
  SMOTE = smote_balance,
  BorderlineSMOTE = borderline_smote_balance,
  SET = set_balance  # SMOTE-ENN+Tomek缩写为SET（符合论文表S9命名）
)

# 创建输出文件夹（按平衡方法分类）
for (method_name in names(balance_methods)) {
  output_dir <- paste0("processed_data/balanced_data/", method_name)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
}

# 存储平衡效果摘要
balance_summary <- data.frame()

# 循环处理每次拆分
for (i in 1:n_splits) {
  # 加载第i次拆分的训练集
  train_data <- read.csv(paste0(split_dir, "/wave5_train_split", i, ".csv"), stringsAsFactors = FALSE)
  
  # 仅保留选中的特征和结局变量（减少冗余）
  train_data_subset <- train_data %>%
    select(all_of(c(selected_features, outcome_var)))
  
  # 计算平衡前的基本信息
  pre_n <- nrow(train_data_subset)
  pre_pos_rate <- round(mean(train_data_subset[[outcome_var]]) * 100, 1)
  
  # 应用三种平衡方法并保存
  for (method_name in names(balance_methods)) {
    balance_fun <- balance_methods[[method_name]]
    balanced_data <- balance_fun(train_data_subset, selected_features, outcome_var)
    
    # 计算平衡后的基本信息
    post_n <- nrow(balanced_data)
    post_pos_rate <- round(mean(balanced_data[[outcome_var]]) * 100, 1)
    
    # 保存平衡后的数据集
    output_path <- paste0(
      "processed_data/balanced_data/", method_name, "/wave5_train_split", i, "_", method_name, ".csv"
    )
    write.csv(balanced_data, output_path, row.names = FALSE)
    
    # 记录摘要信息
    summary_row <- data.frame(
      split_id = i,
      balance_method = method_name,
      pre_balance_n = pre_n,
      pre_positive_rate = pre_pos_rate,
      post_balance_n = post_n,
      post_positive_rate = post_pos_rate
    )
    balance_summary <- rbind(balance_summary, summary_row)
    
    # 打印进度信息
    cat(paste("=== 第", i, "次拆分 -", method_name, "平衡完成 ===\n"))
    cat(paste("平衡前：样本量", pre_n, "，焦虑患病率", pre_pos_rate, "%\n"))
    cat(paste("平衡后：样本量", post_n, "，焦虑患病率", post_pos_rate, "%\n\n"))
  }
}

# --------------------------
# Step 4: 保存平衡效果摘要（供论文表S9参考）
summary_dir <- "processed_data/balanced_data/summary"
if (!dir.exists(summary_dir)) {
  dir.create(summary_dir, recursive = TRUE)
}

write.csv(
  balance_summary,
  paste0(summary_dir, "/balance_effect_summary.csv"),
  row.names = FALSE
)

# --------------------------
# Step 5: 保存关键中间数据（供后续建模直接调用）
# 1. 保存选中特征+结局变量的外部验证数据集
wave4_validation_subset <- wave4_encoded %>%
  select(all_of(c(selected_features, outcome_var)))

write.csv(
  wave4_validation_subset,
  "processed_data/balanced_data/wave4_validation_subset.csv",
  row.names = FALSE
)

# 2. 保存每种方法的平衡后数据摘要（按方法分组）
method_summary <- balance_summary %>%
  group_by(balance_method) %>%
  summarise(
    mean_pre_n = mean(pre_balance_n),
    mean_pre_pos_rate = mean(pre_positive_rate),
    mean_post_n = mean(post_balance_n),
    mean_post_pos_rate = mean(post_positive_rate),
    sd_post_pos_rate = sd(post_positive_rate),
    .groups = "drop"
  )

write.csv(
  method_summary,
  paste0(summary_dir, "/method_balance_summary.csv"),
  row.names = FALSE
)

# --------------------------
# 最终验证信息打印
cat("\n=== 数据平衡处理完成 ===\n")
cat("输出文件说明：\n")
cat("1. processed_data/balanced_data/[方法名]/: 10次拆分的平衡后训练集\n")
cat("2. processed_data/balanced_data/summary/: 平衡效果摘要表\n")
cat("3. processed_data/balanced_data/wave4_validation_subset.csv: 外部验证子集\n")
cat("\n各方法平衡后平均焦虑患病率：\n")
for (method_name in names(balance_methods)) {
  rate <- filter(method_summary, balance_method == method_name)$mean_post_pos_rate
  cat(paste("-", method_name, ":", round(rate, 1), "%\n"))
}
cat("\n下一步：运行05_model_training.R训练模型（使用SET方法平衡后的数据效果最优）\n")