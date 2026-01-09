# 05_model_training.R: 模型训练（5个基础模型 + Stacking集成模型）
# 依赖包（需提前安装：install.packages(c("glmnet", "class", "randomForest", "e1071", "nnet", "caretEnsemble", "dplyr", "tidyr"))）
library(glmnet)         # LR模型（v4.1-8）
library(class)          # kNN模型（v7.3-22）
library(randomForest)   # RF模型（v4.7-1.1）
library(e1071)          # SVM模型（v1.7-13）
library(nnet)           # NN模型（v7.3-19）
library(caretEnsemble)  # Stacking集成（v2.0.1）
library(dplyr)          # 数据操作（v1.1.4）
library(tidyr)          # 数据整理（v1.3.0）

# --------------------------
# Step 1: 加载数据（来自前序脚本输出）
# 1. 加载选中的特征列表
feature_file <- "processed_data/feature_selection/final_selected_features.csv"
selected_features <- read.csv(feature_file)$selected_variable
outcome_var <- "anxiety"  # 结局变量（0=无焦虑，1=有焦虑）

# 2. 加载10次拆分的测试集（用于内部验证）
split_dir <- "processed_data/splits"
test_sets <- list()
for (i in 1:10) {
  test_sets[[i]] <- read.csv(paste0(split_dir, "/wave5_test_split", i, ".csv"), stringsAsFactors = FALSE) %>%
    select(all_of(c(selected_features, outcome_var)))
}

# 3. 加载平衡后的训练集（优先使用SET方法，论文中SET效果最优）
balance_method <- "SET"  # 可选：SMOTE、BorderlineSMOTE、SET
balanced_train_dir <- paste0("processed_data/balanced_data/", balance_method)
if (!dir.exists(balanced_train_dir)) {
  stop(paste("请先运行04_data_balancing.R生成", balance_method, "方法平衡后的训练集！"))
}

balanced_train_sets <- list()
for (i in 1:10) {
  balanced_train_sets[[i]] <- read.csv(
    paste0(balanced_train_dir, "/wave5_train_split", i, "_", balance_method, ".csv"),
    stringsAsFactors = FALSE
  ) %>%
    select(all_of(c(selected_features, outcome_var)))
}

# 4. 加载外部验证数据集（无需平衡，直接用于验证）
external_val_data <- read.csv("processed_data/balanced_data/wave4_validation_subset.csv", stringsAsFactors = FALSE) %>%
  select(all_of(c(selected_features, outcome_var)))

# --------------------------
# Step 2: 定义模型训练函数（严格遵循论文Table S4和Section 2.5参数）
# 模型1: Logistic Regression (LR-SET)
train_lr <- function(train_data, selected_features, outcome_var) {
  x <- model.matrix(~ . - 1, data = train_data[, selected_features, drop = FALSE])
  y <- train_data[[outcome_var]]
  
  # 论文参数：family=binomial, α=0.6, λ=0.012, maxit=1000
  set.seed(3031)
  lr_model <- glmnet(
    x = x,
    y = y,
    family = "binomial",
    alpha = 0.6,
    lambda = 0.012,
    maxit = 1000
  )
  
  # 定义预测函数（返回概率和类别）
  predict_fun <- function(model, newdata) {
    newx <- model.matrix(~ . - 1, data = newdata[, selected_features, drop = FALSE])
    prob <- predict(model, newx = newx, type = "response")
    class <- ifelse(prob > 0.5, 1, 0)
    return(list(prob = as.vector(prob), class = class))
  }
  
  return(list(model = lr_model, predict_fun = predict_fun))
}

# 模型2: k-Nearest Neighbors (kNN-SET)
train_knn <- function(train_data, selected_features, outcome_var) {
  x_train <- train_data[, selected_features, drop = FALSE]
  y_train <- train_data[[outcome_var]]
  
  # 论文参数：k=5, 距离=Euclidean, 权重=Uniform
  knn_params <- list(
    k = 5,
    distance = "Euclidean"
  )
  
  # 定义预测函数
  predict_fun <- function(model, newdata) {
    x_new <- newdata[, selected_features, drop = FALSE]
    # 标准化数据（kNN对尺度敏感）
    x_train_scaled <- scale(model$x_train)
    x_new_scaled <- scale(x_new, center = attr(x_train_scaled, "scaled:center"), scale = attr(x_train_scaled, "scaled:scale"))
    
    class <- knn(
      train = x_train_scaled,
      test = x_new_scaled,
      cl = model$y_train,
      k = knn_params$k,
      prob = TRUE
    )
    # 提取预测概率（少数类的概率）
    prob <- attr(class, "prob")
    # 调整概率方向（确保1是焦虑类的概率）
    prob <- ifelse(as.integer(as.character(class)) == 1, prob, 1 - prob)
    
    return(list(prob = as.vector(prob), class = as.integer(as.character(class))))
  }
  
  return(list(model = list(x_train = x_train, y_train = y_train), predict_fun = predict_fun))
}

# 模型3: Random Forest (RF-SET)
train_rf <- function(train_data, selected_features, outcome_var) {
  x_train <- train_data[, selected_features, drop = FALSE]
  y_train <- as.factor(train_data[[outcome_var]])
  
  # 论文参数：ntree=100, mtry=5, maxdepth=3, minsplit=2
  set.seed(3031)
  rf_model <- randomForest(
    x = x_train,
    y = y_train,
    ntree = 100,
    mtry = 5,
    maxdepth = 3,
    minsplit = 2,
    importance = TRUE,
    proximity = FALSE,
    do.trace = FALSE
  )
  
  # 定义预测函数
  predict_fun <- function(model, newdata) {
    x_new <- newdata[, selected_features, drop = FALSE]
    prob <- predict(model, newdata = x_new, type = "prob")[, "1"]  # 焦虑类（1）的概率
    class <- predict(model, newdata = x_new)
    return(list(prob = as.vector(prob), class = as.integer(as.character(class))))
  }
  
  return(list(model = rf_model, predict_fun = predict_fun))
}

# 模型4: Support Vector Machine (SVM-SET)
train_svm <- function(train_data, selected_features, outcome_var) {
  x_train <- train_data[, selected_features, drop = FALSE]
  y_train <- as.factor(train_data[[outcome_var]])
  
  # 论文参数：kernel=radial, cost=5, epsilon=0.1, tolerance=0.001, maxit=300
  set.seed(3031)
  svm_model <- svm(
    x = x_train,
    y = y_train,
    kernel = "radial",
    cost = 5,
    epsilon = 0.1,
    tolerance = 0.001,
    maxit = 300,
    probability = TRUE  # 启用概率预测
  )
  
  # 定义预测函数
  predict_fun <- function(model, newdata) {
    x_new <- newdata[, selected_features, drop = FALSE]
    class <- predict(model, newdata = x_new)
    prob <- attr(predict(model, newdata = x_new, probability = TRUE), "probabilities")[, "1"]
    return(list(prob = as.vector(prob), class = as.integer(as.character(class))))
  }
  
  return(list(model = svm_model, predict_fun = predict_fun))
}

# 模型5: Neural Network (NN-SET)
train_nn <- function(train_data, selected_features, outcome_var) {
  x_train <- train_data[, selected_features, drop = FALSE]
  y_train <- as.factor(train_data[[outcome_var]])
  
  # 标准化特征（NN对尺度敏感）
  x_train_scaled <- scale(x_train)
  scale_center <- attr(x_train_scaled, "scaled:center")
  scale_scale <- attr(x_train_scaled, "scaled:scale")
  
  # 论文参数：hidden neurons=100, activation=ReLU, solver=Adam, α=0.0001, maxit=200
  # nnet包中用size指定隐藏神经元数，maxit指定迭代次数
  set.seed(3031)
  nn_model <- nnet(
    x = x_train_scaled,
    y = y_train,
    size = 100,  # 隐藏神经元数
    decay = 0.0001,  # L2正则化（对应α）
    maxit = 200,
    trace = FALSE,
    softmax = TRUE  # 二分类用softmax输出概率
  )
  
  # 定义预测函数
  predict_fun <- function(model, newdata) {
    x_new <- newdata[, selected_features, drop = FALSE]
    x_new_scaled <- scale(x_new, center = scale_center, scale = scale_scale)
    prob <- predict(model, newdata = x_new_scaled, type = "raw")[, "1"]
    class <- ifelse(prob > 0.5, 1, 0)
    return(list(prob = as.vector(prob), class = class))
  }
  
  return(list(model = nn_model, predict_fun = predict_fun, scale_info = list(center = scale_center, scale = scale_scale)))
}

# 模型6: Stacking Ensemble (Stacking-SET)
train_stacking <- function(train_data, base_models, selected_features, outcome_var) {
  # 第一步：生成基模型的Out-of-Fold (OOF) 预测（避免数据泄漏）
  set.seed(3031)
  cv_folds <- createFolds(train_data[[outcome_var]], k = 10, returnTrain = TRUE)
  
  # 存储每个基模型的OOF预测
  oof_predictions <- data.frame(
    lr_prob = numeric(nrow(train_data)),
    knn_prob = numeric(nrow(train_data)),
    rf_prob = numeric(nrow(train_data)),
    svm_prob = numeric(nrow(train_data)),
    nn_prob = numeric(nrow(train_data))
  )
  
  # 训练基模型并生成OOF预测
  for (fold in names(cv_folds)) {
    train_idx <- cv_folds[[fold]]
    fold_train <- train_data[train_idx, ]
    fold_val <- train_data[-train_idx, ]
    
    # 训练每个基模型
    fold_lr <- train_lr(fold_train, selected_features, outcome_var)
    fold_knn <- train_knn(fold_train, selected_features, outcome_var)
    fold_rf <- train_rf(fold_train, selected_features, outcome_var)
    fold_svm <- train_svm(fold_train, selected_features, outcome_var)
    fold_nn <- train_nn(fold_train, selected_features, outcome_var)
    
    # 生成OOF预测并存储
    oof_predictions[-train_idx, "lr_prob"] <- fold_lr$predict_fun(fold_lr$model, fold_val)$prob
    oof_predictions[-train_idx, "knn_prob"] <- fold_knn$predict_fun(fold_knn$model, fold_val)$prob
    oof_predictions[-train_idx, "rf_prob"] <- fold_rf$predict_fun(fold_rf$model, fold_val)$prob
    oof_predictions[-train_idx, "svm_prob"] <- fold_svm$predict_fun(fold_svm$model, fold_val)$prob
    oof_predictions[-train_idx, "nn_prob"] <- fold_nn$predict_fun(fold_nn$model, fold_val)$prob
  }
  
  # 第二步：训练元模型（逻辑回归），并应用基模型权重（论文指定：RF=0.35, LR=0.25, NN=0.20, kNN=0.10, SVM=0.10）
  weighted_oof <- oof_predictions %>%
    mutate(
      weighted_prob = rf_prob * 0.35 + lr_prob * 0.25 + nn_prob * 0.20 + knn_prob * 0.10 + svm_prob * 0.10
    )
  
  meta_model <- glm(
    formula = as.formula(paste(outcome_var, "~ weighted_prob")),
    data = cbind(train_data[, outcome_var, drop = FALSE], weighted_oof),
    family = "binomial"
  )
  
  # 第三步：定义Stacking预测函数（先基模型加权，再元模型预测）
  predict_fun <- function(model, base_model_list, newdata) {
    # 基模型预测
    lr_prob <- base_model_list$lr$predict_fun(base_model_list$lr$model, newdata)$prob
    knn_prob <- base_model_list$knn$predict_fun(base_model_list$knn$model, newdata)$prob
    rf_prob <- base_model_list$rf$predict_fun(base_model_list$rf$model, newdata)$prob
    svm_prob <- base_model_list$svm$predict_fun(base_model_list$svm$model, newdata)$prob
    nn_prob <- base_model_list$nn$predict_fun(base_model_list$nn$model, newdata)$prob
    
    # 加权融合
    weighted_prob <- rf_prob * 0.35 + lr_prob * 0.25 + nn_prob * 0.20 + knn_prob * 0.10 + svm_prob * 0.10
    
    # 元模型预测最终概率和类别
    meta_data <- data.frame(weighted_prob = weighted_prob)
    final_prob <- predict(model$meta_model, newdata = meta_data, type = "response")
    final_class <- ifelse(final_prob > 0.5, 1, 0)
    
    return(list(
      prob = as.vector(final_prob),
      class = final_class,
      base_probs = data.frame(lr_prob, knn_prob, rf_prob, svm_prob, nn_prob, weighted_prob)
    ))
  }
  
  return(list(model = list(meta_model = meta_model, oof_predictions = oof_predictions), predict_fun = predict_fun))
}

# --------------------------
# Step 3: 批量训练10次拆分的所有模型
n_splits <- 10
model_list <- list()  # 存储所有模型
prediction_results <- list()  # 存储预测结果

# 创建输出文件夹（按模型和拆分次数分类）
output_dir <- "processed_data/model_training"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# 循环处理每次拆分
for (i in 1:n_splits) {
  cat(paste("=== 第", i, "次拆分 - 模型训练开始 ===\n"))
  train_data <- balanced_train_sets[[i]]
  test_data <- test_sets[[i]]
  
  # 1. 训练5个基础模型
  cat("训练LR模型...\n")
  lr_model <- train_lr(train_data, selected_features, outcome_var)
  
  cat("训练kNN模型...\n")
  knn_model <- train_knn(train_data, selected_features, outcome_var)
  
  cat("训练RF模型...\n")
  rf_model <- train_rf(train_data, selected_features, outcome_var)
  
  cat("训练SVM模型...\n")
  svm_model <- train_svm(train_data, selected_features, outcome_var)
  
  cat("训练NN模型...\n")
  nn_model <- train_nn(train_data, selected_features, outcome_var)
  
  # 存储基础模型列表（用于Stacking）
  base_models <- list(
    lr = lr_model,
    knn = knn_model,
    rf = rf_model,
    svm = svm_model,
    nn = nn_model
  )
  
  # 2. 训练Stacking模型
  cat("训练Stacking模型...\n")
  stacking_model <- train_stacking(train_data, base_models, selected_features, outcome_var)
  
  # 3. 生成内部测试集预测结果
  cat("生成测试集预测结果...\n")
  test_pred <- list(
    lr = lr_model$predict_fun(lr_model$model, test_data),
    knn = knn_model$predict_fun(knn_model$model, test_data),
    rf = rf_model$predict_fun(rf_model$model, test_data),
    svm = svm_model$predict_fun(svm_model$model, test_data),
    nn = nn_model$predict_fun(nn_model$model, test_data),
    stacking = stacking_model$predict_fun(stacking_model$model, base_models, test_data)
  )
  
  # 4. 生成外部验证集预测结果
  cat("生成外部验证集预测结果...\n")
  external_pred <- list(
    lr = lr_model$predict_fun(lr_model$model, external_val_data),
    knn = knn_model$predict_fun(knn_model$model, external_val_data),
    rf = rf_model$predict_fun(rf_model$model, external_val_data),
    svm = svm_model$predict_fun(svm_model$model, external_val_data),
    nn = nn_model$predict_fun(nn_model$model, external_val_data),
    stacking = stacking_model$predict_fun(stacking_model$model, base_models, external_val_data)
  )
  
  # 5. 保存模型和预测结果
  # 保存模型（用saveRDS，便于后续加载）
  model_save_dir <- paste0(output_dir, "/split", i)
  if (!dir.exists(model_save_dir)) {
    dir.create(model_save_dir)
  }
  
  saveRDS(lr_model, paste0(model_save_dir, "/lr_model.rds"))
  saveRDS(knn_model, paste0(model_save_dir, "/knn_model.rds"))
  saveRDS(rf_model, paste0(model_save_dir, "/rf_model.rds"))
  saveRDS(svm_model, paste0(model_save_dir, "/svm_model.rds"))
  saveRDS(nn_model, paste0(model_save_dir, "/nn_model.rds"))
  saveRDS(stacking_model, paste0(model_save_dir, "/stacking_model.rds"))
  
  # 保存预测结果（包含真实标签和各模型预测）
  test_pred_df <- test_data[, outcome_var, drop = FALSE] %>%
    mutate(
      lr_prob = test_pred$lr$prob,
      lr_class = test_pred$lr$class,
      knn_prob = test_pred$knn$prob,
      knn_class = test_pred$knn$class,
      rf_prob = test_pred$rf$prob,
      rf_class = test_pred$rf$class,
      svm_prob = test_pred$svm$prob,
      svm_class = test_pred$svm$class,
      nn_prob = test_pred$nn$prob,
      nn_class = test_pred$nn$class,
      stacking_prob = test_pred$stacking$prob,
      stacking_class = test_pred$stacking$class
    )
  
  external_pred_df <- external_val_data[, outcome_var, drop = FALSE] %>%
    mutate(
      lr_prob = external_pred$lr$prob,
      lr_class = external_pred$lr$class,
      knn_prob = external_pred$knn$prob,
      knn_class = external_pred$knn$class,
      rf_prob = external_pred$rf$prob,
      rf_class = external_pred$rf$class,
      svm_prob = external_pred$svm$prob,
      svm_class = external_pred$svm$class,
      nn_prob = external_pred$nn$prob,
      nn_class = external_pred$nn$class,
      stacking_prob = external_pred$stacking$prob,
      stacking_class = external_pred$stacking$class
    )
  
  write.csv(test_pred_df, paste0(model_save_dir, "/test_pred_results.csv"), row.names = FALSE)
  write.csv(external_pred_df, paste0(model_save_dir, "/external_pred_results.csv"), row.names = FALSE)
  
  # 存储到全局列表
  model_list[[i]] <- list(
    split_id = i,
    base_models = base_models,
    stacking_model = stacking_model
  )
  
  prediction_results[[i]] <- list(
    split_id = i,
    test_pred = test_pred_df,
    external_pred = external_pred_df
  )
  
  cat(paste("第", i, "次拆分 - 模型训练完成\n\n"))
}

# --------------------------
# Step 4: 保存全局结果（供后续06_model_evaluation.R调用）
# 1. 合并10次拆分的预测结果
all_test_pred <- do.call(rbind, lapply(prediction_results, function(x) {
  x$test_pred %>% mutate(split_id = x$split_id)
}))

all_external_pred <- do.call(rbind, lapply(prediction_results, function(x) {
  x$external_pred %>% mutate(split_id = x$split_id)
}))

write.csv(all_test_pred, paste0(output_dir, "/all_test_pred_results.csv"), row.names = FALSE)
write.csv(all_external_pred, paste0(output_dir, "/all_external_pred_results.csv"), row.names = FALSE)

# 2. 保存模型训练摘要
train_summary <- data.frame(
  split_id = 1:n_splits,
  train_sample_size = sapply(balanced_train_sets, nrow),
  test_sample_size = sapply(test_sets, nrow),
  train_anxiety_rate = sapply(balanced_train_sets, function(x) round(mean(x[[outcome_var]]) * 100, 1)),
  test_anxiety_rate = sapply(test_sets, function(x) round(mean(x[[outcome_var]]) * 100, 1))
)

write.csv(train_summary, paste0(output_dir, "/model_training_summary.csv"), row.names = FALSE)

# --------------------------
# 最终验证信息打印
cat("\n=== 所有模型训练完成 ===\n")
cat("输出文件说明：\n")
cat("1. processed_data/model_training/split[1-10]/: 每次拆分的模型文件和预测结果\n")
cat("2. all_test_pred_results.csv: 10次拆分测试集的合并预测结果\n")
cat("3. all_external_pred_results.csv: 10次拆分外部验证集的合并预测结果\n")
cat("4. model_training_summary.csv: 训练数据摘要（样本量、焦虑患病率）\n")
cat("\n下一步：运行06_model_evaluation.R计算模型性能指标（AUC、准确率、F1等）\n")