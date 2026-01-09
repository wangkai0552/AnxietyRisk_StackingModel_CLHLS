# 07_model_interpretation.R: 模型解释（SHAP分析 + LIME分析 + ICE曲线）
# 依赖包（需提前安装：install.packages(c("shapviz", "lime", "dplyr", "tidyr", "ggplot2", "gridExtra", "caret"))）
library(shapviz)      # SHAP值计算与可视化（v0.10.0）
library(lime)         # 局部模型解释（v0.5.3）
library(dplyr)        # 数据操作（v1.1.4）
library(tidyr)        # 数据整理（v1.3.0）
library(ggplot2)      # 可视化（v3.4.4）
library(gridExtra)    # 多图组合（v2.3）
library(caret)        # 数据预处理辅助（v6.0-94）

# --------------------------
# Step 1: 加载数据与模型（来自前序脚本输出）
# 1. 加载模型（优先使用第1次拆分的Stacking-SET模型，论文核心模型）
model_dir <- "processed_data/model_training/split1"
if (!dir.exists(model_dir)) {
  stop("请先运行05_model_training.R生成模型文件！")
}

# 加载Stacking模型（含基模型和预测函数）
stacking_model <- readRDS(paste0(model_dir, "/stacking_model.rds"))
# 加载选中的特征列表
selected_features <- read.csv("processed_data/feature_selection/final_selected_features.csv")$selected_variable
# 加载预处理后的完整数据（用于计算SHAP值）
wave5_encoded <- read.csv("processed_data/wave5_imputed_encoded.csv", stringsAsFactors = FALSE)
wave4_encoded <- read.csv("processed_data/wave4_imputed_encoded.csv", stringsAsFactors = FALSE)
outcome_var <- "anxiety"

# 提取建模用数据（仅保留选中特征和结局变量）
model_data_internal <- wave5_encoded %>%
  select(all_of(c(selected_features, outcome_var))) %>%
  na.omit()  # 移除可能残留的缺失值

model_data_external <- wave4_encoded %>%
  select(all_of(c(selected_features, outcome_var))) %>%
  na.omit()

# 2. 加载预测结果（用于验证）
test_pred <- read.csv(paste0(model_dir, "/test_pred_results.csv"), stringsAsFactors = FALSE)
external_pred <- read.csv(paste0(model_dir, "/external_pred_results.csv"), stringsAsFactors = FALSE)

# --------------------------
# Step 2: SHAP分析（全局特征重要性 + 特征影响方向，对应论文图5、Table S11、S14）
# 定义SHAP分析函数（适配Stacking模型）
calculate_shap <- function(model, model_data, selected_features, outcome_var) {
  # 提取特征矩阵（不含结局变量）
  x_matrix <- model_data %>%
    select(all_of(selected_features)) %>%
    as.matrix()
  
  # 定义Stacking模型的预测概率函数（shapviz要求输入函数返回概率）
  predict_prob_fun <- function(model, newdata) {
    model$predict_fun(model$model, model$base_models, as.data.frame(newdata))$prob
  }
  
  # 计算SHAP值（使用 Kernel SHAP 方法，适配复杂模型）
  set.seed(6061)  # 固定随机种子
  sv <- shapviz(
    object = model,
    X = x_matrix,
    model_type = "classification",
    predict_function = predict_prob_fun,
    class_idx = 1  # 预测焦虑类（1）的概率
  )
  
  # 提取SHAP值数据框
  shap_df <- sv$shap_values %>%
    as.data.frame() %>%
    mutate(
      sample_id = 1:n(),
      actual_class = model_data[[outcome_var]],
      predicted_prob = predict_prob_fun(model, x_matrix)
    ) %>%
    relocate(sample_id, actual_class, predicted_prob)
  
  # 计算特征重要性（平均绝对SHAP值）
  feature_importance <- sv$importance %>%
    as.data.frame() %>%
    mutate(feature = rownames(.)) %>%
    rename(mean_abs_shap = 1) %>%
    arrange(desc(mean_abs_shap))
  
  return(list(
    shap_viz = sv,
    shap_df = shap_df,
    feature_importance = feature_importance
  ))
}

# 计算内部队列（Wave5）的SHAP值
shap_internal <- calculate_shap(stacking_model, model_data_internal, selected_features, outcome_var)
# 计算外部队列（Wave4）的SHAP值
shap_external <- calculate_shap(stacking_model, model_data_external, selected_features, outcome_var)

# 2.1 绘制SHAP核心可视化图（对应论文图5）
# 图5A：SHAP摘要图（特征影响方向 + 特征值大小）
shap_summary_plot <- sv_importance(shap_internal$shap_viz, kind = "beeswarm") +
  labs(x = "Impact on Model Output (SHAP Value)", y = "Feature Importance (Ranked by Mean |SHAP|)") +
  theme_bw() +
  theme(legend.position = "bottom", axis.text.y = element_text(size = 10))

# 图5B：特征重要性条形图（平均绝对SHAP值）
shap_bar_plot <- sv_importance(shap_internal$shap_viz, kind = "bar") +
  labs(x = "Mean Absolute SHAP Value", y = "Feature") +
  theme_bw() +
  theme(axis.text.y = element_text(size = 10))

# 组合图5并保存
fig5 <- gridExtra::grid.arrange(shap_summary_plot, shap_bar_plot, ncol = 2, widths = c(1.2, 1))
ggsave(paste0("processed_data/model_interpretation/SHAP_Feature_Analysis.pdf"), 
       fig5, width = 16, height = 8, dpi = 300)

# 2.2 输出特征重要性结果（对应Table S11、S14）
# 内部队列特征重要性
shap_importance_internal <- shap_internal$feature_importance %>%
  mutate(cohort = "Internal (Wave5)")

# 外部队列特征重要性
shap_importance_external <- shap_external$feature_importance %>%
  mutate(cohort = "External (Wave4)")

# 跨队列特征重要性对比（对应Table S14）
shap_cross_cohort <- shap_importance_internal %>%
  select(feature, mean_abs_shap) %>%
  rename(internal_shap = mean_abs_shap) %>%
  left_join(
    shap_importance_external %>% select(feature, mean_abs_shap) %>% rename(external_shap = mean_abs_shap),
    by = "feature"
  ) %>%
  mutate(
    spearman_corr = cor(internal_shap, external_shap, method = "spearman"),
    p_value = cor.test(internal_shap, external_shap, method = "spearman")$p.value
  )

# 保存SHAP结果
write.csv(
  shap_importance_internal,
  "processed_data/model_interpretation/SHAP_Importance_Internal.csv",
  row.names = FALSE
)
write.csv(
  shap_importance_external,
  "processed_data/model_interpretation/SHAP_Importance_External.csv",
  row.names = FALSE
)
write.csv(
  shap_cross_cohort,
  "processed_data/model_interpretation/SHAP_Cross_Cohort_Comparison.csv",
  row.names = FALSE
)

# --------------------------
# Step 3: LIME分析（局部模型解释，对应论文图7）
# 针对核心特征“body pain level”进行局部解释
lime_analysis <- function(model, model_data, selected_features, outcome_var, target_feature = "body_pain_level") {
  # 准备LIME用数据（特征矩阵 + 标签）
  lime_data <- model_data %>%
    select(all_of(c(selected_features, outcome_var)))
  
  # 定义LIME的预测函数（返回概率矩阵：列1=非焦虑，列2=焦虑）
  lime_predict_fun <- function(model, newdata) {
    prob_anxiety <- model$predict_fun(model$model, model$base_models, newdata)$prob
    cbind(1 - prob_anxiety, prob_anxiety)  # 输出两列概率
  }
  
  # 训练LIME解释器
  set.seed(6061)
  lime_explainer <- lime(
    x = lime_data %>% select(-all_of(outcome_var)),
    model = model,
    predict_function = lime_predict_fun,
    bin_continuous = TRUE,  # 连续变量分箱
    n_bins = 5
  )
  
  # 选择20个代表性样本（覆盖不同焦虑状态和目标特征值）
  sample_idx <- sample(1:nrow(lime_data), size = 20, replace = FALSE)
  lime_samples <- lime_data[sample_idx, ] %>% select(-all_of(outcome_var))
  
  # 解释每个样本的预测结果（聚焦目标特征）
  lime_explanations <- explain(
    x = lime_samples,
    explainer = lime_explainer,
    n_features = 5,  # 每个样本展示5个关键特征
    n_permutations = 1000,  # 置换次数
    feature_select = "auto"
  )
  
  # 绘制LIME图（对应论文图7：身体疼痛对预测结果的局部影响）
  lime_plot <- plot_explanations(lime_explanations, feature_name = target_feature) +
    labs(x = "Instances Ordered by Predicted Anxiety Risk", y = "LIME Value (Impact on Anxiety Risk)") +
    theme_bw() +
    theme(legend.position = "bottom")
  
  return(list(
    lime_explanations = lime_explanations,
    lime_plot = lime_plot
  ))
}

# 执行LIME分析（目标特征：body_pain_level）
lime_result <- lime_analysis(stacking_model, model_data_internal, selected_features, outcome_var)

# 保存LIME结果和图表
ggsave(paste0("processed_data/model_interpretation/LIME_BodyPain_Analysis.pdf"),
       lime_result$lime_plot, width = 12, height = 6, dpi = 300)

write.csv(
  lime_result$lime_explanations %>% as.data.frame(),
  "processed_data/model_interpretation/LIME_Explanations.csv",
  row.names = FALSE
)

# --------------------------
# Step 4: ICE曲线（个体条件期望曲线，对应论文图6）
# 定义ICE曲线绘制函数（固定其他特征，展示单个特征的影响）
plot_ice_curves <- function(model, model_data, selected_features, outcome_var, target_features) {
  # 准备数据：对每个目标特征，生成不同取值，固定其他特征为均值/众数
  ice_data_list <- lapply(target_features, function(feature) {
    # 提取特征类型（序数/二分类）
    feature_vals <- unique(model_data[[feature]]) %>% sort()
    # 固定其他特征（数值型取均值，分类变量取众数）
    fixed_data <- model_data %>%
      select(-all_of(c(feature, outcome_var))) %>%
      summarise(across(everything(), 
                       ~ifelse(is.numeric(.), mean(., na.rm = TRUE), names(which.max(table(.))))))
    
    # 生成不同特征值的数据集
    expand_data <- lapply(feature_vals, function(val) {
      cbind(fixed_data, !!feature := val) %>%
        mutate(feature_value = val, target_feature = feature)
    }) %>% do.call(rbind, .)
    
    # 预测每个组合的焦虑概率
    expand_data$predicted_prob <- stacking_model$predict_fun(
      stacking_model$model,
      stacking_model$base_models,
      expand_data %>% select(-feature_value, -target_feature)
    )$prob
    
    # 标记焦虑状态（基于原始数据的结局变量）
    expand_data$anxiety_status <- ifelse(
      model_data[[outcome_var]][1] == 1, "Anxiety Risk (GAD-7≥5)", "Non-Anxiety (GAD-7<5)"
    )
    
    return(expand_data)
  }) %>% do.call(rbind, .)
  
  # 绘制ICE曲线（按目标特征分面）
  ice_plot <- ggplot(ice_data_list, aes(x = feature_value, y = predicted_prob, color = anxiety_status, group = anxiety_status)) +
    geom_line(linewidth = 1.2) +
    facet_wrap(~target_feature, scales = "free_x") +
    labs(x = "Feature Value", y = "Predicted Anxiety Probability", color = "Group") +
    theme_bw() +
    theme(legend.position = "bottom", strip.text = element_text(size = 11))
  
  return(ice_plot)
}

# 选择论文中的4个核心特征绘制ICE曲线
target_features_ice <- c("body_pain_level", "self_rated_health", "agricultural_work", "chronic_disease_total")
ice_curve_plot <- plot_ice_curves(stacking_model, model_data_internal, selected_features, outcome_var, target_features_ice)

# 保存ICE曲线（对应论文图6）
ggsave(paste0("processed_data/model_interpretation/ICE_Curves.pdf"),
       ice_curve_plot, width = 14, height = 10, dpi = 300)

# --------------------------
# Step 5: 输出综合解释结果（支撑论文讨论）
# 1. 核心特征影响总结（对应论文3.5节）
feature_impact_summary <- shap_internal$feature_importance %>%
  head(10) %>%  # 取Top10特征
  left_join(
    model_data_internal %>%
      summarise(across(all_of(selected_features), ~cor(., get(outcome_var), use = "complete.obs"))) %>%
      pivot_longer(everything(), names_to = "feature", values_to = "corr_with_anxiety"),
    by = "feature"
  ) %>%
  mutate(
    impact_direction = ifelse(corr_with_anxiety > 0, "Positive (Increases Anxiety Risk)", "Negative (Decreases Anxiety Risk)"),
    feature_name_cn = case_when(
      feature == "body_pain_level" ~ "Body Pain Level",
      feature == "sleep_duration" ~ "Sleep Duration",
      feature == "self_rated_health" ~ "Self-rated Health Status",
      feature == "hearing_problem" ~ "Hearing Problem",
      feature == "air_quality_satisfaction" ~ "Air Quality Satisfaction",
      feature == "BADL_disability" ~ "BADL Disability",
      feature == "MMSE_status" ~ "MMSE Status",
      feature == "TICS_status" ~ "TICS Status",
      feature == "children_satisfaction" ~ "Children Satisfaction",
      feature == "government_pension" ~ "Government Pension",
      TRUE ~ feature
    )
  ) %>%
  select(feature_name_cn, mean_abs_shap, corr_with_anxiety, impact_direction)

# 保存特征影响总结
write.csv(
  feature_impact_summary,
  "processed_data/model_interpretation/Feature_Impact_Summary.csv",
  row.names = FALSE
)

# --------------------------
# 最终验证信息打印
cat("\n=== 模型解释分析完成 ===\n")
cat("输出文件说明：\n")
cat("1. processed_data/model_interpretation/SHAP_Feature_Analysis.pdf: SHAP摘要图+特征重要性条形图（论文图5）\n")
cat("2. processed_data/model_interpretation/LIME_BodyPain_Analysis.pdf: 身体疼痛的LIME局部解释图（论文图7）\n")
cat("3. processed_data/model_interpretation/ICE_Curves.pdf: 4个核心特征的ICE曲线（论文图6）\n")
cat("4. SHAP_Importance_Internal/External.csv: 内/外部队列SHAP特征重要性（Table S11/S14）\n")
cat("5. Feature_Impact_Summary.csv: 核心特征影响方向总结（支撑论文讨论）\n")
cat("\n关键结果验证：\n")
cat(paste("Top5核心特征（按SHAP重要性）:", paste(head(shap_internal$feature_importance$feature, 5), collapse = ", "), "\n"))
cat(paste("身体疼痛与焦虑的相关性:", round(feature_impact_summary$corr_with_anxiety[1], 3), "\n"))
cat(paste("跨队列特征重要性Spearman相关系数:", round(shap_cross_cohort$spearman_corr[1], 3), "\n"))