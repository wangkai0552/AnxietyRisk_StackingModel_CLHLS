# 06_model_evaluation.R: 模型性能评估（指标计算+可视化）
# 依赖包（需提前安装：install.packages(c("pROC", "caret", "dplyr", "tidyr", "ggplot2", "gridExtra", "broom"))）
library(pROC)         # ROC曲线与AUC计算（v1.18.4）
library(caret)        # 分类性能指标（v6.0-94）
library(dplyr)        # 数据操作（v1.1.4）
library(tidyr)        # 数据整理（v1.3.0）
library(ggplot2)      # 可视化（v3.4.4）
library(gridExtra)    # 多图组合（v2.3）
library(broom)        # 统计结果整理（v1.0.5）

# --------------------------
# Step 1: 加载数据（来自05_model_training.R的输出）
# 1. 加载预测结果（内部测试集+外部验证集）
model_dir <- "processed_data/model_training"
if (!dir.exists(model_dir)) {
  stop("请先运行05_model_training.R生成预测结果！")
}

# 内部测试集预测结果（10次拆分合并）
all_test_pred <- read.csv(paste0(model_dir, "/all_test_pred_results.csv"), stringsAsFactors = FALSE)
# 外部验证集预测结果（10次拆分合并，取平均概率）
all_external_pred <- read.csv(paste0(model_dir, "/all_external_pred_results.csv"), stringsAsFactors = FALSE)

# 定义结局变量和模型名称
outcome_var <- "anxiety"
model_names <- c("LR", "kNN", "RF", "SVM", "NN", "Stacking")
model_cols <- list(
  LR = c("lr_prob", "lr_class"),
  kNN = c("knn_prob", "knn_class"),
  RF = c("rf_prob", "rf_class"),
  SVM = c("svm_prob", "svm_class"),
  NN = c("nn_prob", "nn_class"),
  Stacking = c("stacking_prob", "stacking_class")
)

# 2. 加载训练摘要（用于参考）
train_summary <- read.csv(paste0(model_dir, "/model_training_summary.csv"), stringsAsFactors = FALSE)

# --------------------------
# Step 2: 定义性能指标计算函数（符合论文2.6节方法）
# 函数1：计算单个数据集的性能指标（内部验证用）
calculate_metrics <- function(pred_df, model_col, outcome_var) {
  prob_col <- model_col[1]
  class_col <- model_col[2]
  
  # 真实标签
  y_true <- pred_df[[outcome_var]]
  # 预测概率和类别
  y_prob <- pred_df[[prob_col]]
  y_pred <- pred_df[[class_col]]
  
  # 1. AUC（梯形积分法，符合论文说明）
  roc_obj <- roc(y_true, y_prob, direction = "<", ci = TRUE)
  auc_val <- roc_obj$auc
  auc_ci <- roc_obj$ci
  
  # 2. 分类指标（准确率、精确率、召回率、F1-score、MCC）
  confusion_mat <- confusionMatrix(factor(y_pred, levels = c(0, 1)), 
                                  factor(y_true, levels = c(0, 1)),
                                  positive = "1")
  accuracy <- confusion_mat$overall["Accuracy"]
  precision <- confusion_mat$byClass["Precision"]
  recall <- confusion_mat$byClass["Recall"]
  f1 <- confusion_mat$byClass["F1"]
  mcc <- mcc(y_true, y_pred)  # Matthews相关系数
  
  # 处理可能的NA值（如无阳性样本时）
  metrics <- data.frame(
    AUC = ifelse(is.na(auc_val), 0, auc_val),
    AUC_lower = ifelse(is.na(auc_ci[1]), 0, auc_ci[1]),
    AUC_upper = ifelse(is.na(auc_ci[3]), 0, auc_ci[3]),
    Accuracy = ifelse(is.na(accuracy), 0, accuracy),
    Precision = ifelse(is.na(precision), 0, precision),
    Recall = ifelse(is.na(recall), 0, recall),
    F1_score = ifelse(is.na(f1), 0, f1),
    MCC = ifelse(is.na(mcc), 0, mcc)
  )
  
  return(metrics)
}

# 函数2：计算外部验证的指标（含1000次bootstrap的95%CI）
calculate_external_metrics <- function(pred_df, model_col, outcome_var, n_bootstrap = 1000) {
  prob_col <- model_col[1]
  class_col <- model_col[2]
  
  y_true <- pred_df[[outcome_var]]
  y_prob <- pred_df[[prob_col]]
  y_pred <- pred_df[[class_col]]
  
  # 基础指标
  base_metrics <- calculate_metrics(pred_df, model_col, outcome_var)
  
  # Bootstrap抽样计算95%CI
  set.seed(5051)  # 固定随机种子
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    # 有放回抽样
    sample_idx <- sample(1:nrow(pred_df), size = nrow(pred_df), replace = TRUE)
    sample_pred <- pred_df[sample_idx, ]
    calculate_metrics(sample_pred, model_col, outcome_var)
  })
  
  # 合并bootstrap结果并计算CI
  bootstrap_df <- do.call(rbind, bootstrap_results)
  ci_metrics <- data.frame(
    AUC = base_metrics$AUC,
    AUC_CI = paste0(round(base_metrics$AUC, 3), " [", round(quantile(bootstrap_df$AUC, 0.025), 3), "–", round(quantile(bootstrap_df$AUC, 0.975), 3), "]"),
    Accuracy = base_metrics$Accuracy,
    Accuracy_CI = paste0(round(base_metrics$Accuracy, 3), " [", round(quantile(bootstrap_df$Accuracy, 0.025), 3), "–", round(quantile(bootstrap_df$Accuracy, 0.975), 3), "]"),
    Precision = base_metrics$Precision,
    Precision_CI = paste0(round(base_metrics$Precision, 3), " [", round(quantile(bootstrap_df$Precision, 0.025), 3), "–", round(quantile(bootstrap_df$Precision, 0.975), 3), "]"),
    Recall = base_metrics$Recall,
    Recall_CI = paste0(round(base_metrics$Recall, 3), " [", round(quantile(bootstrap_df$Recall, 0.025), 3), "–", round(quantile(bootstrap_df$Recall, 0.975), 3), "]"),
    F1_score = base_metrics$F1_score,
    F1_CI = paste0(round(base_metrics$F1_score, 3), " [", round(quantile(bootstrap_df$F1_score, 0.025), 3), "–", round(quantile(bootstrap_df$F1_score, 0.975), 3), "]"),
    MCC = base_metrics$MCC,
    MCC_CI = paste0(round(base_metrics$MCC, 3), " [", round(quantile(bootstrap_df$MCC, 0.025), 3), "–", round(quantile(bootstrap_df$MCC, 0.975), 3), "]")
  )
  
  return(ci_metrics)
}

# 函数3：绘制校准曲线（对比预测概率与实际发生率）
plot_calibration_curve <- function(pred_df, model_cols, outcome_var) {
  # 按预测概率分10组
  calib_data <- pred_df %>%
    mutate(
      # 为每个模型添加概率分组
      across(all_of(sapply(model_cols, function(x) x[1])), 
             ~cut(., breaks = seq(0, 1, 0.1), include.lowest = TRUE),
             .names = "{.col}_bin")
    ) %>%
    pivot_longer(
      cols = c(all_of(sapply(model_cols, function(x) x[1])), all_of(sapply(model_cols, function(x) x[1]) %>% paste0("_bin"))),
      names_to = c("model", ".value"),
      names_pattern = "(.*)_(prob|bin)"
    ) %>%
    filter(!is.na(bin)) %>%
    group_by(model, bin) %>%
    summarise(
      mean_prob = mean(prob, na.rm = TRUE),
      actual_rate = mean(get(outcome_var), na.rm = TRUE),
      n = n(),
      .groups = "drop"
    ) %>%
    mutate(model = factor(model, levels = names(model_cols), labels = paste0(names(model_cols), "-SET")))
  
  # 绘制校准曲线
  ggplot(calib_data, aes(x = mean_prob, y = actual_rate, color = model, group = model)) +
    geom_point(aes(size = n)) +
    geom_line(linewidth = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    labs(x = "Predicted Anxiety Probability", 
         y = "Actual Anxiety Incidence Rate",
         color = "Model",
         size = "Sample Size per Bin") +
    theme_bw() +
    theme(legend.position = "bottom")
}

# --------------------------
# Step 3: 内部验证结果计算（10次拆分的均值±标准差）
# 按拆分次数分组计算
internal_metrics_list <- lapply(unique(all_test_pred$split_id), function(split) {
  split_data <- filter(all_test_pred, split_id == split)
  lapply(names(model_cols), function(model) {
    metrics <- calculate_metrics(split_data, model_cols[[model]], outcome_var)
    data.frame(
      split_id = split,
      model = paste0(model, "-SET"),
      metrics
    )
  }) %>% do.call(rbind, .)
}) %>% do.call(rbind, .)

# 汇总内部验证结果（均值±标准差）
internal_summary <- internal_metrics_list %>%
  group_by(model) %>%
  summarise(
    AUC_mean = mean(AUC),
    AUC_SD = sd(AUC),
    Accuracy_mean = mean(Accuracy),
    Accuracy_SD = sd(Accuracy),
    Precision_mean = mean(Precision),
    Precision_SD = sd(Precision),
    Recall_mean = mean(Recall),
    Recall_SD = sd(Recall),
    F1_mean = mean(F1_score),
    F1_SD = sd(F1_score),
    MCC_mean = mean(MCC),
    MCC_SD = sd(MCC),
    .groups = "drop"
  ) %>%
  # 格式化输出（均值±标准差）
  mutate(
    AUC = paste0(round(AUC_mean, 3), "±", round(AUC_SD, 3)),
    Accuracy = paste0(round(Accuracy_mean, 3), "±", round(Accuracy_SD, 3)),
    Precision = paste0(round(Precision_mean, 3), "±", round(Precision_SD, 3)),
    Recall = paste0(round(Recall_mean, 3), "±", round(Recall_SD, 3)),
    F1_score = paste0(round(F1_mean, 3), "±", round(F1_SD, 3)),
    MCC = paste0(round(MCC_mean, 3), "±", round(MCC_SD, 3))
  ) %>%
  select(model, AUC, Accuracy, Precision, Recall, F1_score, MCC)

# 打印内部验证结果（对应论文Table S8）
cat("=== 内部验证（10次拆分）性能汇总 ===\n")
print(internal_summary, row.names = FALSE)

# --------------------------
# Step 4: 外部验证结果计算（点估计+95%CI）
# 外部验证集取10次拆分的平均概率（减少随机误差）
external_avg_pred <- all_external_pred %>%
  group_by(across(-c(split_id, ends_with("_prob"), ends_with("_class")))) %>%
  summarise(
    lr_prob = mean(lr_prob),
    lr_class = ifelse(mean(lr_prob) > 0.5, 1, 0),
    knn_prob = mean(knn_prob),
    knn_class = ifelse(mean(knn_prob) > 0.5, 1, 0),
    rf_prob = mean(rf_prob),
    rf_class = ifelse(mean(rf_prob) > 0.5, 1, 0),
    svm_prob = mean(svm_prob),
    svm_class = ifelse(mean(svm_prob) > 0.5, 1, 0),
    nn_prob = mean(nn_prob),
    nn_class = ifelse(mean(nn_prob) > 0.5, 1, 0),
    stacking_prob = mean(stacking_prob),
    stacking_class = ifelse(mean(stacking_prob) > 0.5, 1, 0),
    .groups = "drop"
  )

# 计算各模型外部验证指标
external_summary <- lapply(names(model_cols), function(model) {
  calculate_external_metrics(external_avg_pred, model_cols[[model]], outcome_var) %>%
    mutate(model = paste0(model, "-SET"))
}) %>% do.call(rbind, .) %>%
  select(model, AUC, AUC_CI, Accuracy, Accuracy_CI, Precision, Precision_CI, Recall, Recall_CI, F1_score, F1_CI, MCC, MCC_CI)

# 打印外部验证结果（对应论文Table S13）
cat("\n=== 外部验证（Wave4）性能汇总 ===\n")
print(external_summary, row.names = FALSE)

# --------------------------
# Step 5: 绘制可视化图表（对应论文图2-4）
# 创建输出文件夹
plot_dir <- "processed_data/model_plots"
if (!dir.exists(plot_dir)) {
  dir.create(plot_dir, recursive = TRUE)
}

# 图2：ROC曲线（所有模型对比）
roc_plot <- ggplot() +
  # 为每个模型添加ROC曲线
  lapply(names(model_cols), function(model) {
    roc_obj <- roc(all_test_pred[[outcome_var]], all_test_pred[[model_cols[[model]][1]]], direction = "<")
    roc_df <- data.frame(
      fpr = 1 - roc_obj$specificities,
      tpr = roc_obj$sensitivities,
      model = paste0(model, "-SET")
    )
    geom_line(data = roc_df, aes(x = fpr, y = tpr, color = model), linewidth = 1.2)
  }) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  labs(x = "False Positive Rate (1-Specificity)", 
       y = "True Positive Rate (Sensitivity)",
       color = "Model") +
  theme_bw() +
  theme(legend.position = "bottom")

# 图3：校准曲线（所有模型对比）
calib_plot <- plot_calibration_curve(all_test_pred, model_cols, outcome_var)

# 图4：精确率-召回率曲线 + 提升曲线
# 精确率-召回率曲线
pr_plot <- all_test_pred %>%
  pivot_longer(
    cols = all_of(sapply(model_cols, function(x) x[1])),
    names_to = "model",
    values_to = "prob"
  ) %>%
  mutate(model = factor(model, levels = sapply(model_cols, function(x) x[1]), 
                        labels = paste0(names(model_cols), "-SET"))) %>%
  ggplot(aes(x = recall, y = precision, color = model)) +
  stat_precision_recall(aes(probability = prob, truth = factor(get(outcome_var))), 
                       data = ., geom = "line", linewidth = 1.2) +
  labs(x = "Recall", y = "Precision", color = "Model") +
  theme_bw() +
  theme(legend.position = "bottom")

# 提升曲线（按预测概率排序，计算提升值=模型检测率/总体患病率）
overall_prevalence <- mean(all_test_pred[[outcome_var]])
lift_plot <- all_test_pred %>%
  pivot_longer(
    cols = all_of(sapply(model_cols, function(x) x[1])),
    names_to = "model",
    values_to = "prob"
  ) %>%
  group_by(model) %>%
  arrange(desc(prob), .by_group = TRUE) %>%
  mutate(
    cumulative_prop = row_number() / n(),
    cumulative_pos = cumsum(get(outcome_var)) / sum(get(outcome_var)),
    lift = cumulative_pos / cumulative_prop
  ) %>%
  ungroup() %>%
  mutate(model = factor(model, levels = sapply(model_cols, function(x) x[1]), 
                        labels = paste0(names(model_cols), "-SET"))) %>%
  ggplot(aes(x = cumulative_prop, y = lift, color = model)) +
  geom_line(linewidth = 1.2) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
  labs(x = "Cumulative Proportion of Samples", y = "Lift Value", color = "Model") +
  theme_bw() +
  theme(legend.position = "bottom")

# 组合图4（精确率-召回率曲线 + 提升曲线）
fig4 <- gridExtra::grid.arrange(pr_plot + theme(legend.position = "none"), 
                                lift_plot + theme(legend.position = "none"),
                                ncol = 2,
                                bottom = gridExtra::grid.legend(pr_plot$labels$color, 
                                                               levels(pr_plot$data$model),
                                                               ncol = 3,
                                                               title = "Model"))

# 保存所有图表（高清PDF格式）
ggsave(paste0(plot_dir, "/ROC_Curve.pdf"), roc_plot, width = 10, height = 8, dpi = 300)
ggsave(paste0(plot_dir, "/Calibration_Curve.pdf"), calib_plot, width = 10, height = 8, dpi = 300)
ggsave(paste0(plot_dir, "/Precision_Recall_Lift_Curves.pdf"), fig4, width = 12, height = 6, dpi = 300)

cat("\n=== 可视化图表已保存至", plot_dir, "===\n")

# --------------------------
# Step 6: 保存结果文件（对应论文补充表格）
# 创建结果文件夹
result_dir <- "processed_data/model_results"
if (!dir.exists(result_dir)) {
  dir.create(result_dir, recursive = TRUE)
}

# 1. 内部验证详细结果（10次拆分）
write.csv(internal_metrics_list, paste0(result_dir, "/internal_validation_detailed.csv"), row.names = FALSE)

# 2. 内部验证汇总（对应Table S8）
write.csv(internal_summary, paste0(result_dir, "/internal_validation_summary.csv"), row.names = FALSE)

# 3. 外部验证汇总（对应Table S13）
write.csv(external_summary, paste0(result_dir, "/external_validation_summary.csv"), row.names = FALSE)

# 4. 模型性能对比表（对应Table S9）
model_comparison <- internal_summary %>%
  left_join(external_summary %>% select(model, AUC_CI, Accuracy_CI, F1_CI), by = "model") %>%
  rename(
    Internal_AUC = AUC,
    Internal_Accuracy = Accuracy,
    Internal_F1 = F1_score,
    External_AUC = AUC_CI,
    External_Accuracy = Accuracy_CI,
    External_F1 = F1_CI
  ) %>%
  select(model, Internal_AUC, External_AUC, Internal_Accuracy, External_Accuracy, Internal_F1, External_F1)

write.csv(model_comparison, paste0(result_dir, "/model_performance_comparison.csv"), row.names = FALSE)

# --------------------------
# 最终验证信息打印
cat("\n=== 模型性能评估完成 ===\n")
cat("输出文件说明：\n")
cat("1. processed_data/model_results/: 性能指标汇总表（内部+外部验证）\n")
cat("2. processed_data/model_plots/: ROC曲线、校准曲线、精确率-召回率+提升曲线\n")
cat("\n关键结果验证（需与论文一致）：\n")
cat(paste("Stacking-SET 内部验证AUC:", filter(internal_summary, model == "Stacking-SET")$AUC, "\n"))
cat(paste("Stacking-SET 外部验证AUC:", filter(external_summary, model == "Stacking-SET")$AUC_CI, "\n"))
cat(paste("模型性能排名（内部验证AUC）:", paste(arrange(internal_summary, desc(substr(Internal_AUC, 1, 5)))$model, collapse = " > "), "\n"))