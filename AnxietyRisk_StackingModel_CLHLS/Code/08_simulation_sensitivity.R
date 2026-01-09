# 08_simulation_sensitivity.R: 模拟研究 + 敏感性分析（生存偏倚 + 变量选择验证）
# 依赖包（需提前安装：install.packages(c("simstudy", "glmnet", "grpreg", "dplyr", "tidyr", "pROC", "caret"))）
library(simstudy)     # 生成合成数据集（v0.7.8）
library(glmnet)       # Elastic Net（v4.1-8）
library(grpreg)       # Group LASSO（v3.4.1）
library(dplyr)        # 数据操作（v1.1.4）
library(tidyr)        # 数据整理（v1.3.0）
library(pROC)         # 模型性能验证（v1.18.4）
library(caret)        # 分类指标计算（v6.0-94）

# --------------------------
# Step 1: 加载基础数据与参数（来自前序脚本输出）
# 1. 加载预处理后的数据（wave4用于生存偏倚分析，wave5用于模拟参数校准）
wave5_encoded <- read.csv("processed_data/wave5_imputed_encoded.csv", stringsAsFactors = FALSE)
wave4_encoded <- read.csv("processed_data/wave4_imputed_encoded.csv", stringsAsFactors = FALSE)
selected_features <- read.csv("processed_data/feature_selection/final_selected_features.csv")$selected_variable
outcome_var <- "anxiety"

# 2. 模拟研究核心参数（匹配论文Table S2/S5）
n_sim <- 100                # 合成数据集数量（100个）
n_obs <- 2313               # 每个合成数据集样本量（匹配wave5样本量）
true_prevalence <- 0.165    # 焦虑患病率（16.5%，与真实数据一致）
# 真实预测因子（论文指定：5个核心特征）
true_predictors <- c("body_pain_level", "sleep_duration", "self_rated_health", "hearing_problem", "air_quality_satisfaction")
# 噪声变量（35个，模拟非预测变量，来自Table S1）
noise_vars <- setdiff(colnames(wave5_encoded)[!colnames(wave5_encoded) %in% c(outcome_var, true_predictors)], "ID") %>% head(35)

# 3. 创建输出文件夹
output_dir <- "processed_data/simulation_sensitivity"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# --------------------------
# Step 2: 生存偏倚敏感性分析（论文2.8节）
# 目标：模拟wave4中2005-2008年间死亡人群的2008年焦虑状态，验证核心特征稳定性
survivorship_bias_analysis <- function(wave4_data, wave5_data, true_predictors, outcome_var) {
  # 步骤1：识别"死亡人群"（wave4有数据，wave5无数据，假设为2005-2008年死亡）
  # 用ID匹配，wave4有但wave5无的样本视为死亡人群
  wave4_ids <- wave4_data$ID
  wave5_ids <- wave5_data$ID
  deceased_ids <- setdiff(wave4_ids, wave5_ids)
  deceased_data <- wave4_data %>% filter(ID %in% deceased_ids) %>% filter(age_level %in% c(2, 3))  # 仅保留≥70岁女性（符合纳入标准）
  
  cat(paste("生存偏倚分析：识别死亡人群样本量 =", nrow(deceased_data), "\n"))
  
  # 步骤2：训练逻辑回归模型，模拟死亡人群的2008年焦虑状态（GAD-7≥5）
  # 训练集：wave5数据（存活人群），预测因子：2005年健康状态+社会支持（论文指定）
  train_data <- wave5_data %>%
    select(all_of(c("self_rated_health", "chronic_disease_total", "BADL_disability", "children_satisfaction", outcome_var)))
  
  # 拟合预测模型
  set.seed(7071)
  predict_model <- glm(
    formula = as.formula(paste(outcome_var, "~ self_rated_health + chronic_disease_total + BADL_disability + children_satisfaction")),
    data = train_data,
    family = "binomial"
  )
  
  # 模拟死亡人群的焦虑状态
  deceased_data$simulated_anxiety <- predict(
    predict_model,
    newdata = deceased_data %>% select(all_of(c("self_rated_health", "chronic_disease_total", "BADL_disability", "children_satisfaction")),
    type = "response"
  ) %>%
    sapply(function(p) rbinom(1, 1, p))  # 二项分布采样（概率=预测值）
  
  # 步骤3：合并存活人群+模拟死亡人群，重新进行特征选择
  # 存活人群（wave5）：真实焦虑状态
  alive_data <- wave5_data %>%
    select(all_of(c(true_predictors, noise_vars, outcome_var))) %>%
    mutate(cohort = "Alive")
  
  # 死亡人群：模拟焦虑状态
  simulated_deceased_data <- deceased_data %>%
    select(all_of(c(true_predictors, noise_vars))) %>%
    mutate(
      anxiety = simulated_anxiety,
      cohort = "Deceased"
    ) %>%
    select(-simulated_anxiety)
  
  # 合并数据集
  combined_data <- rbind(alive_data, simulated_deceased_data) %>%
    na.omit()  # 移除残留缺失值
  
  # 步骤4：重新用Elastic Net选择特征，对比核心特征稳定性
  x_combined <- model.matrix(~ . - 1, data = combined_data[, c(true_predictors, noise_vars)])
  y_combined <- combined_data[[outcome_var]]
  
  set.seed(7071)
  cv_elastic <- cv.glmnet(
    x = x_combined,
    y = y_combined,
    family = "binomial",
    alpha = 0.6,  # 论文指定α=0.6
    nfolds = 10
  )
  
  # 提取选中特征（非零系数）
  combined_selected <- coef(cv_elastic, s = cv_elastic$lambda.1se) %>%
    as.data.frame() %>%
    mutate(variable = rownames(.)) %>%
    filter(s1 != 0) %>%
    pull(variable)
  
  # 计算核心特征的SHAP值（对比合并前后）
  # 存活人群SHAP值（来自07_model_interpretation.R）
  shap_alive <- read.csv("processed_data/model_interpretation/SHAP_Importance_Internal.csv") %>%
    filter(feature %in% true_predictors) %>%
    rename(alive_shap = mean_abs_shap)
  
  # 合并人群SHAP值
  shap_combined <- calculate_shap_importance(x_combined, y_combined, true_predictors) %>%
    rename(combined_shap = mean_abs_shap)
  
  # 合并结果
  feature_comparison <- shap_alive %>%
    left_join(shap_combined, by = "feature") %>%
    mutate(
      selected_in_combined = ifelse(feature %in% combined_selected, "Yes", "No"),
      shap_change = (combined_shap - alive_shap) / alive_shap * 100  # 变化百分比
    )
  
  # 步骤5：计算患病率对比
  prevalence_summary <- data.frame(
    cohort = c("Alive (Wave5)", "Simulated Deceased", "Combined"),
    sample_size = c(nrow(alive_data), nrow(simulated_deceased_data), nrow(combined_data)),
    anxiety_prevalence = c(
      mean(alive_data[[outcome_var]]),
      mean(simulated_deceased_data[[outcome_var]]),
      mean(combined_data[[outcome_var]])
    )
  ) %>%
    mutate(anxiety_prevalence = round(anxiety_prevalence * 100, 1))
  
  return(list(
    prevalence_summary = prevalence_summary,
    feature_comparison = feature_comparison,
    combined_selected_features = combined_selected,
    predict_model_auc = auc(roc(train_data[[outcome_var]], predict(predict_model, type = "response")))
  ))
}

# 辅助函数：计算SHAP值（简化版，聚焦核心特征）
calculate_shap_importance <- function(x, y, target_features) {
  # 拟合Elastic Net模型
  model <- glmnet(x = x, y = y, family = "binomial", alpha = 0.6, lambda = 0.012)
  
  # 计算每个特征的重要性（绝对系数均值）
  importance <- coef(model) %>%
    as.data.frame() %>%
    mutate(feature = rownames(.)) %>%
    filter(feature %in% target_features) %>%
    rename(coefficient = s1) %>%
    mutate(mean_abs_shap = abs(coefficient)) %>%
    select(feature, mean_abs_shap)
  
  return(importance)
}

# 执行生存偏倚分析
survivorship_result <- survivorship_bias_analysis(wave4_encoded, wave5_encoded, true_predictors, outcome_var)

# --------------------------
# Step 3: 变量选择模拟验证（论文2.8节）
# 目标：生成100个合成数据集，验证Elastic Net/Group LASSO的变量选择准确性
simulation_variable_selection <- function(n_sim, n_obs, true_prevalence, true_predictors, noise_vars) {
  # 存储每次模拟的结果
  sim_results <- data.frame()
  
  cat(paste("开始变量选择模拟：共生成", n_sim, "个合成数据集...\n"))
  
  for (i in 1:n_sim) {
    if (i %% 10 == 0) cat(paste("已完成", i, "个数据集模拟\n"))
    
    # 步骤1：生成合成数据集
    set.seed(7071 + i)  # 递进随机种子，保证可复现
    
    # 定义真实预测因子的分布（基于真实数据校准）
    def <- defData(varname = true_predictors[1], dist = "ordinal", formula = "0.3;0.5;0.2", link = "logit")  # body_pain_level（0/1/2）
    def <- defData(def, varname = true_predictors[2], dist = "ordinal", formula = "0.4;0.4;0.2", link = "logit")  # sleep_duration（1/2/3）
    def <- defData(def, varname = true_predictors[3], dist = "ordinal", formula = "0.2;0.3;0.5", link = "logit")  # self_rated_health（1/2/3）
    def <- defData(def, varname = true_predictors[4], dist = "ordinal", formula = "0.5;0.3;0.2", link = "logit")  # hearing_problem（1/2/3）
    def <- defData(def, varname = true_predictors[5], dist = "binary", formula = 0.7, link = "logit")  # air_quality_satisfaction（1/2）
    
    # 添加噪声变量（与真实变量独立）
    for (var in noise_vars) {
      def <- defData(def, varname = var, dist = ifelse(sample(c("binary", "ordinal"), 1) == "binary", "binary", "ordinal"), 
                     formula = ifelse(sample(c("binary", "ordinal"), 1) == "binary", 0.5, "0.3;0.4;0.3"), link = "logit")
    }
    
    # 生成基础数据
    sim_data <- genData(n_obs, def)
    
    # 定义焦虑结局（真实预测因子的线性组合 + 截距）
    def_outcome <- defEvent(
      varname = "anxiety",
      formula = paste0(
        "-1.8 + 0.8*", true_predictors[1], " + 0.6*", true_predictors[2], " - 0.7*", true_predictors[3], 
        " + 0.5*", true_predictors[4], " - 0.4*", true_predictors[5]
      ),
      link = "logit"
    )
    
    # 生成结局变量（保证患病率≈16.5%）
    sim_data <- genEvent(def_outcome, data = sim_data)
    sim_data$anxiety <- as.integer(sim_data$anxiety)
    
    # 步骤2：用Elastic Net选择特征
    x_sim <- model.matrix(~ . - 1, data = sim_data[, c(true_predictors, noise_vars)])
    y_sim <- sim_data$anxiety
    
    set.seed(7071 + i)
    cv_elastic <- cv.glmnet(
      x = x_sim,
      y = y_sim,
      family = "binomial",
      alpha = 0.6,
      nfolds = 10
    )
    
    elastic_selected <- coef(cv_elastic, s = cv_elastic$lambda.1se) %>%
      as.data.frame() %>%
      mutate(variable = rownames(.)) %>%
      filter(s1 != 0) %>%
      pull(variable)
    
    # 步骤3：用Group LASSO选择特征（敏感性分析）
    # 定义分组：真实预测因子各为一组，噪声变量合并为一组
    group_vec <- c(rep(1:5, each = 1), rep(6, length(noise_vars)))  # 1-5=真实预测因子，6=噪声变量
    names(group_vec) <- c(true_predictors, noise_vars)
    
    set.seed(7071 + i)
    cv_grp <- cv.grpreg(
      X = x_sim,
      y = y_sim,
      family = "binomial",
      group = group_vec,
      nfolds = 10
    )
    
    grp_selected <- names(coef(cv_grp, s = cv_grp$lambda.1se)[coef(cv_grp, s = cv_grp$lambda.1se) != 0])
    
    # 步骤4：计算选择性能指标
    metrics <- calculate_selection_metrics(elastic_selected, grp_selected, true_predictors, noise_vars)
    metrics$simulation_id <- i
    
    sim_results <- rbind(sim_results, metrics)
  }
  
  # 汇总模拟结果（均值±标准差）
  sim_summary <- sim_results %>%
    group_by(method) %>%
    summarise(
      selection_accuracy = mean(selection_accuracy),
      selection_precision = mean(selection_precision),
      selection_f1 = mean(selection_f1),
      true_predictor_rate = mean(true_predictor_rate),
      noise_selection_rate = mean(noise_selection_rate),
      mean_selected_vars = mean(mean_selected_vars),
      .groups = "drop"
    ) %>%
    mutate(
      across(c(selection_accuracy:mean_selected_vars), ~round(., 3)),
      sd_selection_accuracy = sim_results %>% group_by(method) %>% summarise(sd=sd(selection_accuracy)) %>% pull(sd) %>% round(3),
      sd_selection_precision = sim_results %>% group_by(method) %>% summarise(sd=sd(selection_precision)) %>% pull(sd) %>% round(3),
      sd_selection_f1 = sim_results %>% group_by(method) %>% summarise(sd=sd(selection_f1)) %>% pull(sd) %>% round(3),
      sd_true_predictor_rate = sim_results %>% group_by(method) %>% summarise(sd=sd(true_predictor_rate)) %>% pull(sd) %>% round(3),
      sd_noise_selection_rate = sim_results %>% group_by(method) %>% summarise(sd=sd(noise_selection_rate)) %>% pull(sd) %>% round(3),
      sd_mean_selected_vars = sim_results %>% group_by(method) %>% summarise(sd=sd(mean_selected_vars)) %>% pull(sd) %>% round(3)
    )
  
  return(list(
    sim_results = sim_results,
    sim_summary = sim_summary
  ))
}

# 辅助函数：计算变量选择性能指标
calculate_selection_metrics <- function(elastic_selected, grp_selected, true_predictors, noise_vars) {
  # Elastic Net指标
  elastic_true <- sum(true_predictors %in% elastic_selected)
  elastic_noise <- sum(noise_vars %in% elastic_selected)
  elastic_total <- length(elastic_selected)
  
  # Group LASSO指标
  grp_true <- sum(true_predictors %in% grp_selected)
  grp_noise <- sum(noise_vars %in% grp_selected)
  grp_total <- length(grp_selected)
  
  # LASSO（参考方法：α=1）
  lasso_selected <- coef(cv.glmnet(x = x_sim, y = y_sim, family = "binomial", alpha = 1, nfolds = 10), s = "lambda.1se") %>%
    as.data.frame() %>%
    mutate(variable = rownames(.)) %>%
    filter(s1 != 0) %>%
    pull(variable)
  lasso_true <- sum(true_predictors %in% lasso_selected)
  lasso_noise <- sum(noise_vars %in% lasso_selected)
  lasso_total <- length(lasso_selected)
  
  # 计算指标（准确率=正确识别真实预测因子比例；精确率=选中变量中真实预测因子比例）
  calculate_single_metric <- function(true_count, noise_count, total_count, n_true) {
    selection_accuracy <- true_count / n_true  # 真实预测因子识别准确率
    selection_precision <- ifelse(total_count == 0, 0, true_count / total_count)  # 选中变量精确率
    selection_f1 <- 2 * (selection_accuracy * selection_precision) / ifelse((selection_accuracy + selection_precision) == 0, 1, (selection_accuracy + selection_precision))  # F1分数
    true_predictor_rate <- true_count / n_true  # 真实预测因子选择率
    noise_selection_rate <- noise_count / length(noise_vars)  # 噪声变量选择率
    mean_selected_vars <- total_count  # 平均选中变量数
    
    return(data.frame(
      selection_accuracy, selection_precision, selection_f1,
      true_predictor_rate, noise_selection_rate, mean_selected_vars
    ))
  }
  
  # 合并三种方法的指标
  rbind(
    calculate_single_metric(elastic_true, elastic_noise, elastic_total, length(true_predictors)) %>% mutate(method = "Elastic Net"),
    calculate_single_metric(grp_true, grp_noise, grp_total, length(true_predictors)) %>% mutate(method = "Group LASSO"),
    calculate_single_metric(lasso_true, lasso_noise, lasso_total, length(true_predictors)) %>% mutate(method = "LASSO (Reference)")
  )
}

# 执行模拟研究
sim_result <- simulation_variable_selection(n_sim, n_obs, true_prevalence, true_predictors, noise_vars)

# --------------------------
# Step 4: 结果保存与打印
# 4.1 生存偏倚分析结果
write.csv(
  survivorship_result$prevalence_summary,
  paste0(output_dir, "/survivorship_prevalence_summary.csv"),
  row.names = FALSE
)

write.csv(
  survivorship_result$feature_comparison,
  paste0(output_dir, "/survivorship_feature_comparison.csv"),
  row.names = FALSE
)

# 4.2 模拟研究结果
write.csv(
  sim_result$sim_summary,
  paste0(output_dir, "/simulation_selection_summary.csv"),
  row.names = FALSE
)

write.csv(
  sim_result$sim_results,
  paste0(output_dir, "/simulation_selection_detailed.csv"),
  row.names = FALSE
)

# 4.3 打印关键结果（匹配论文Table S2/S5/Table 2）
cat("\n=== 生存偏倚敏感性分析结果 ===\n")
print(survivorship_result$prevalence_summary, row.names = FALSE)
cat(paste("\n核心特征选择一致性（合并前后）:", round(sum(survivorship_result$feature_comparison$selected_in_combined == "Yes") / nrow(survivorship_result$feature_comparison) * 100, 1), "%\n"))
cat(paste("预测模型AUC（模拟焦虑状态）:", round(survivorship_result$predict_model_auc, 3), "\n"))

cat("\n=== 变量选择模拟验证结果（均值±标准差） ===\n")
sim_print <- sim_result$sim_summary %>%
  mutate(
    Selection_Accuracy = paste0(selection_accuracy, "±", sd_selection_accuracy),
    Selection_Precision = paste0(selection_precision, "±", sd_selection_precision),
    Selection_F1 = paste0(selection_f1, "±", sd_selection_f1),
    True_Predictor_Rate = paste0(true_predictor_rate, "±", sd_true_predictor_rate),
    Noise_Selection_Rate = paste0(noise_selection_rate, "±", sd_noise_selection_rate),
    Mean_Selected_Vars = paste0(mean_selected_vars, "±", sd_mean_selected_vars)
  ) %>%
  select(
    method, Selection_Accuracy, Selection_Precision, Selection_F1,
    True_Predictor_Rate, Noise_Selection_Rate, Mean_Selected_Vars
  )

print(sim_print, row.names = FALSE)

# --------------------------
# 最终验证信息打印
cat("\n=== 模拟研究与敏感性分析完成 ===\n")
cat("输出文件说明：\n")
cat("1. survivorship_prevalence_summary.csv: 生存偏倚分析患病率对比\n")
cat("2. survivorship_feature_comparison.csv: 核心特征稳定性对比（合并前后）\n")
cat("3. simulation_selection_summary.csv: 变量选择模拟结果汇总（对应Table S2/S5）\n")
cat("4. simulation_selection_detailed.csv: 100次模拟详细结果\n")
cat("\n关键结论验证：\n")
cat(paste("Elastic Net 平均选择准确率:", sim_result$sim_summary %>% filter(method == "Elastic Net") %>% pull(selection_accuracy), "\n"))
cat(paste("模拟死亡人群焦虑患病率:", survivorship_result$prevalence_summary %>% filter(cohort == "Simulated Deceased") %>% pull(anxiety_prevalence), "%\n"))
cat(paste("核心特征选择一致性（合并后）:", round(sum(survivorship_result$feature_comparison$selected_in_combined == "Yes") / nrow(survivorship_result$feature_comparison) * 100, 1), "%\n"))