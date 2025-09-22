# Package Check/Load
packages <- c("caret","pROC","ggplot2","here","tidyverse","glmnet","randomForest","e1071","kernlab","networkD3", "ggalluvial", "scales", "ggtext","stringr")
options(repos=c(CRAN="https://cloud.r-project.org"))
for (p in packages){if (!require(p, character.only=TRUE)){
  install.packages(p, dependencies=TRUE)
  library(p, character.only=TRUE)}}

set.seed(42)

#=============================================
# Model Features (p-value < 0.05 + AUC > 0.65) 
#=============================================

# File path
data_file <- here("Input", "Dataset.csv")
out_dir <- "Results"
if (!dir.exists(out_dir)) dir.create(out_dir)

# Load dataset
data <- read.csv(data_file)
X <- data %>% select(-Class,-SampleID)
X <- X %>% mutate(across(everything(), ~as.numeric(as.character(.x))))
SampleID <- data$SampleID
y_all <- factor(data$Class)

# 1. Filter by RSD in QCs
if ("QC" %in% levels(y_all)) {
  qc_index <- which(y_all == "QC")
  if (length(qc_index) > 0) {
    rsd_qc <- sapply(X[qc_index, , drop=FALSE], function(col) {
      m <- mean(col, na.rm=TRUE)
      s <- sd(col, na.rm=TRUE)
      if (is.na(m) || m == 0) return(NA)
      return((s / m) * 100)
    })
    keep_rsd <- which(rsd_qc <= 25)  # RSD <= 25%
    X <- X[, keep_rsd, drop=FALSE]
  }
}


# Remove QC samples
non_qc_index <- which(y_all != "QC")
X_noQC <- X[non_qc_index, , drop=FALSE]
y_noQC <- factor(y_all[non_qc_index], levels = c("Control","Case"))
ID_noQC <- SampleID[non_qc_index]

# QCs based on y_all
qc_index_all <- which(y_all == "QC")
X_QC_raw <- if (length(qc_index_all) > 0) X[qc_index_all, , drop=FALSE] else X[0, , drop=FALSE]
Label_QC <- if (length(qc_index_all) > 0) y_all[qc_index_all] else factor(character(0), levels="QC")

# Train/Test split
train_index <- createDataPartition(y_noQC, p=0.80, list=FALSE)  # 80/20 split
X_train_raw <- X_noQC[train_index, ]
y_train <- y_noQC[train_index]
ID_train <- ID_noQC[train_index]

X_val_raw <- X_noQC[-train_index, ]
y_val <- y_noQC[-train_index]
ID_val <- ID_noQC[-train_index]

global_min <- suppressWarnings(min(as.matrix(X_train_raw), na.rm = TRUE))

# 2. Filter by missingness within train groups (Case/Control)
missing_rate_by_group <- function(df, labels, thr=0.3) {
  groups <- levels(labels)[levels(labels) %in% c("Control","Case")]
  bad_feats <- c()
  for (feat in colnames(df)) {
    for (g in groups) {
      idx <- which(labels == g)
      miss_rate <- mean(is.na(df[idx, feat]))
      if (miss_rate > thr) {
        bad_feats <- c(bad_feats, feat)
        break
      }
    }
  }
  return(setdiff(colnames(df), bad_feats))
}

feats_keep <- missing_rate_by_group(X_train_raw, y_train, thr=0.3)
X_train_raw <- X_train_raw[, feats_keep, drop=FALSE]
X_val_raw   <- X_val_raw[, feats_keep, drop=FALSE]

if (nrow(X_QC_raw) > 0) {
  X_QC_raw <- X_QC_raw[, feats_keep, drop=FALSE]
}

# 1/5 min value feature (all missing)
X_train_imp <- X_train_raw %>%
  mutate(across(everything(), ~ {
    mn <- suppressWarnings(min(.x, na.rm = TRUE))
    if (!is.finite(mn)) mn <- global_min   # use global minimum if column is 100% NA
    ifelse(is.na(.x), mn/5, .x)
  }))

X_val_imp <- X_val_raw %>%
  mutate(across(everything(), ~ {
    mn <- suppressWarnings(min(X_train_raw[[cur_column()]], na.rm = TRUE))
    if (!is.finite(mn)) mn <- global_min   # use global minimum if column is 100% NA
    ifelse(is.na(.x), mn/5, .x)
  }))

X_QC_imp <- NULL
if (nrow(X_QC_raw) > 0) {
  X_QC_imp <- X_QC_raw %>%
    mutate(across(everything(), ~ { 
      mn <- suppressWarnings(min(X_train_raw[[cur_column()]], na.rm = TRUE))
      if (!is.finite(mn)) mn <- global_min   # fallback - 100% NA
      ifelse(is.na(.x), mn/5, .x)
    }))}

if (!is.null(X_QC_imp) && nrow(X_QC_imp) > 0){
  X_QC_imp <- X_QC_imp[, colnames(X_train_imp), drop=FALSE]
}
Normalize_train_val <- function(X_train, X_val, X_QC=NULL){
  
  # Row sums
  train_row_sums <- rowSums(X_train, na.rm = TRUE)
  X_train_norm <- sweep(X_train, 1, train_row_sums, FUN = "/") |> as.data.frame()
  mean_train_sum <- mean(train_row_sums)
  val_row_sums <- rowSums(X_val, na.rm = TRUE)
  X_val_norm <- sweep(X_val, 1, val_row_sums / mean_train_sum, FUN = "/") |> as.data.frame()
  
  
  # Pseudocount
  pseudocount <- apply(X_train_norm, 2, min, na.rm = TRUE) / 2
  
  # Log transform
  log_transform <- function(df, pseudocount) {
    sweep(df, 2, pseudocount, FUN = "+") |>
      log10() |>
      as.data.frame()
  }
  
  X_train_log <- log_transform(X_train_norm, pseudocount)
  X_val_log   <- log_transform(X_val_norm, pseudocount)
  
  # Center/scale (Pareto) com parâmetros do treino
  train_center <- colMeans(X_train_log, na.rm = TRUE)
  train_scale  <- sqrt(apply(X_train_log, 2, sd, na.rm = TRUE))
  val_center <- colMeans(X_val_log, na.rm = TRUE)
  val_scale  <- sqrt(apply(X_val_log, 2, sd, na.rm = TRUE))
  
  pareto_scale <- function(df, center, scale) {
    scaled <- scale(df, center = center, scale = scale)
    return(as.data.frame(scaled))
  }
  
  X_train_final <- pareto_scale(X_train_log, train_center, train_scale)
  X_val_final   <- pareto_scale(X_val_log,   val_center, val_scale)
  
  X_QC_final <- NULL
  if (!is.null(X_QC) && nrow(X_QC) > 0){
    X_QC_norm <- sweep(X_QC, 1, rowSums(X_QC, na.rm=TRUE), "/") |> as.data.frame()
    X_QC_log  <- log_transform(X_QC_norm, pseudocount)
    
    # Calcular center/scale do próprio QC
    qc_center <- colMeans(X_QC_log, na.rm=TRUE)
    qc_scale  <- sqrt(apply(X_QC_log, 2, sd, na.rm=TRUE))
    
    X_QC_final <- pareto_scale(X_QC_log, qc_center, qc_scale)
  }
  
  list(train=X_train_final, val=X_val_final, QC=X_QC_final)
}

Processed <- Normalize_train_val(X_train_imp, X_val_imp,X_QC_imp)

X_train <- Processed$train
X_val   <- Processed$val
X_QC    <- Processed$QC


# Define feature selection
select_features <- function(X, y){
  # Ensure y is a factor
  y <- factor(y, levels = c("Control", "Case"))
  
  # Remove constant features
  non_constant <- sapply(X, function(col){
    v <- var(col, na.rm=TRUE)
    if (is.na(v) || v == 0) return(FALSE)
    return(TRUE)
  })
  
  # If ALL columns are constant, return all
  if (sum(non_constant) == 0) return(names(X))
  X <- X[, non_constant, drop=FALSE]
  
  # Check that y has both classes
  if (length(unique(y)) < 2) return(names(X))
  
  # T test + AUC filter
  pvals <- sapply(X, function(feat){
    if (length(unique(feat)) < 2) return(NA) # skip constants
    tryCatch(t.test(feat ~ y)$p.value, error = function(e) NA)
  })
  
  aucs <- apply(X, 2, function(feat){
    roc_obj <- tryCatch(
      roc(y, feat, quiet=TRUE, levels=c("Control","Case")),
      error=function(e) return(NA)
    )
    if (!is.null(roc_obj)) auc(roc_obj) else NA
  })
  
  keep <- which(!is.na(pvals) & !is.na(aucs) & pvals <= 0.05 & aucs >=0.65)
  if (length(keep) == 0) keep <- 1:ncol(X) # Keep all if none pass
  
  return(names(X)[keep])
}

# Models to compare
model_list <- list(
  Logistic = "glm",
  ElasticNet = "glmnet",
  SVM_Linear = "svmLinear",
  SVM_Radial = "svmRadial",
  RandomForest = "rf"
)
results_summary = list()

# CV Handler
ctrl <- trainControl(
  method="cv",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Iterate through models
for (model_name in names(model_list)){
  cat("\n--Running: ", model_name, "...\n")
  method <- model_list[[model_name]]
  
  # Filter features on train folds, fit model, predict on test
  folds <- createFolds(y_train, k=5, returnTrain=TRUE)
  all_preds <- data.frame( ID = ID_train, obs=y_train, pred_prob=NA, fold=NA)
  
  for (i in seq_along(folds)){
    train_idx <- folds[[i]]
    test_idx <- setdiff(seq_along(y_train), train_idx)
    
    X_tr <- X_train[train_idx, ]
    y_tr <- y_train[train_idx]
    X_te <- X_train[test_idx, ]
    y_te <- y_train[test_idx]
    
    # Features on train only
    feats <- select_features(X_tr, y_tr)
    X_tr <- X_tr[, feats, drop=FALSE]
    X_te <- X_te[, feats, drop=FALSE]
    
    # Fit Model
    fit <- train(x = X_tr, 
                 y=y_tr,
                 method=method,
                 metric="ROC",
                 trControl=ctrl,
                 tuneLength=5)
    
    # Predict
    prob <- predict(fit, newdata=X_te, type="prob")[, "Case"]
    all_preds$pred_prob[test_idx] <- prob
    all_preds$fold[test_idx] <- i
  }
  
  # Calc CV performance
  all_preds$pred <- factor(ifelse(all_preds$pred_prob > 0.5, "Case", "Control"),
                           levels = c("Control","Case"))
  cm_cv <- confusionMatrix(all_preds$pred, all_preds$obs, positive = "Case")
  roc_cv <- roc(all_preds$obs, all_preds$pred_prob, levels = c("Control", "Case"), positive = "Case")
  
  metrics_cv <- data.frame(
    Model = model_name,
    Dataset = "CV",
    Accuracy = round(cm_cv$overall["Accuracy"], 3),
    Sensitivity = round(cm_cv$byClass["Sensitivity"], 3),
    Specificity = round(cm_cv$byClass["Specificity"], 3),
    Precision = round(cm_cv$byClass["Precision"], 3),
    F1_Score = round(2 * (cm_cv$byClass["Precision"] * cm_cv$byClass["Sensitivity"])/(cm_cv$byClass["Precision"] + cm_cv$byClass["Sensitivity"]), 3),
    AUC = round(auc(roc_cv), 3)
  )
  
  # Hold-out Validation
  feats_val <- select_features(X_train, y_train)
  X_tr_final <- X_train[, feats_val, drop=FALSE]
  X_val_final <- X_val[, feats_val, drop=FALSE]
  
  final_fit <- train(x = X_tr_final,
                     y = y_train,
                     method = method,
                     metric = "ROC",
                     trControl = ctrl,
                     tuneLength = 5)
  
  prob_val <- predict(final_fit, newdata = X_val_final, type="prob")[, "Case"]
  pred_val <- factor(ifelse(prob_val > 0.5, "Case", "Control"), levels = c("Control", "Case"))
  
  cm_val <- confusionMatrix(pred_val, y_val, positive = "Case")
  roc_val <- roc(y_val, prob_val, levels = c("Control","Case"),  positive = "Case")
  
  metrics_val <- data.frame(
    Model = model_name,
    Dataset = "Validation",
    Accuracy     = round(cm_val$overall["Accuracy"], 3),
    Sensitivity  = round(cm_val$byClass["Sensitivity"], 3),
    Specificity  = round(cm_val$byClass["Specificity"], 3),
    Precision    = round(cm_val$byClass["Precision"], 3),
    F1_Score     = round(2 * (cm_val$byClass["Precision"] * cm_val$byClass["Sensitivity"])/(cm_val$byClass["Precision"] + cm_val$byClass["Sensitivity"]), 3),
    AUC          = round(auc(roc_val), 3)
  )
  roc_dir <- here("Results", "ROC")
  dir.create(roc_dir, recursive = TRUE, showWarnings = FALSE)
  png(file.path(roc_dir, paste0("ROC_", model_name, ".png")), width = 1600, height = 1200, res = 300)
  plot(1 - roc_cv$specificities, roc_cv$sensitivities, type = "l", col = "blue", lwd = 2,
       xlab = "False Positive Rate", ylab = "Sensitivity",
       main = paste("ROC -", model_name))
  lines(1 - roc_val$specificities, roc_val$sensitivities, col = "red", lwd = 2)
  legend("bottomright", legend = c(
    paste0("Training AUC = ", round(auc(roc_cv), 3)),
    paste0("Validation AUC = ", round(auc(roc_val),3))), col = c("blue","red"), lwd = 2, cex = 0.9)
  dev.off()
  
  # Collect results
  results_summary[[model_name]] <- rbind(metrics_cv, metrics_val)
}

# Save results
all_results <- bind_rows(results_summary)
Table_dir <- here("Results", "Table")
dir.create(Table_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(all_results, file.path(Table_dir,"Model_Performance_Summary.csv"), row.names=FALSE)

# Compute compact feature table with p-value, AUC, and Fold Change
compute_feature_stats <- function(X_norm, X_raw, y){
  y <- factor(y, levels=c("Control","Case"))
  
  # Remove constant features
  non_constant <- sapply(X_norm, function(col){
    v <- var(col, na.rm=TRUE)
    if (is.na(v) || v == 0) return(FALSE)
    return(TRUE)
  })
  X_norm <- X_train[, non_constant, drop=FALSE]
  X_raw  <- X_train_imp[, colnames(X_norm), drop=FALSE] 
  
  # Initialize vectors
  pvals <- numeric(ncol(X_norm))
  aucs  <- numeric(ncol(X_norm))
  fcs   <- numeric(ncol(X_norm))
  
  for (i in seq_along(X_norm)){
    feat_norm <- X_norm[[i]]
    feat_raw  <- X_raw[[i]]
    
    # p-value
    if(length(unique(feat_norm)) < 2){
      pvals[i] <- NA
    } else {
      pvals[i] <- tryCatch(t.test(feat_norm ~ y)$p.value, error=function(e) NA)
    }
    
    # AUC
    roc_obj <- tryCatch(roc(y, feat_norm, quiet=TRUE, levels=c("Control","Case")),
                        error=function(e) NA)
    aucs[i] <- ifelse(!is.na(roc_obj), auc(roc_obj), NA)
    
    # Fold change
    case_mean <- mean(feat_raw[y == "Case"], na.rm=TRUE)
    ctrl_mean <- mean(feat_raw[y == "Control"], na.rm=TRUE)
    if (is.finite(case_mean) && is.finite(ctrl_mean) && ctrl_mean != 0){
      fcs[i] <- case_mean / ctrl_mean
    } else {
      fcs[i] <- NA
    }
  }
  
  # Select features passing thresholds
  keep <- which(!is.na(pvals) & !is.na(aucs) & pvals <= 0.05 & aucs >= 0.65)
  
  # Create compact table
  df <- data.frame(
    Feature   = names(X_norm)[keep],
    p_value   = round(pvals[keep], 4),
    AUC       = round(aucs[keep], 3),
    FoldChange = round(fcs[keep], 3)
  )
  
  # Optional: sort by AUC descending
  df <- df[order(-df$AUC), ]
  
  return(df)
}


# Apply to training set
features_table <- compute_feature_stats(X_train,X_train_imp, y_train)

# Save CSV
write.csv(features_table, file.path(Table_dir, "Selected_Features_Compact.csv"), row.names = FALSE)

cat("Compact feature table saved: Feature | p_value | AUC\n")

# Compute full feature table with p-value and AUC for all features
compute_feature_stats_all <- function(X_norm, X_imp, y){
  y <- factor(y, levels = c("Control","Case"))
  
  # Remove constant features (apenas no normalizado)
  non_constant <- sapply(X_norm, function(col){
    v <- var(col, na.rm=TRUE)
    if (is.na(v) || v == 0) return(FALSE)
    return(TRUE)
  })
  X_norm <- X_train[, non_constant, drop=FALSE]
  X_imp  <- X_train_imp[, colnames(X_norm), drop=FALSE] 
  
  # Initialize vectors
  pvals <- numeric(ncol(X_norm))
  aucs  <- numeric(ncol(X_norm))
  fcs   <- numeric(ncol(X_norm))
  l2fc  <- numeric(ncol(X_norm))
  dirs  <- character(ncol(X_norm))
  
  for (i in seq_along(X_norm)){
    feat_norm <- X_train[[i]]
    feat_imp  <- X_train_imp[[i]]
    
    # p-value
    if(length(unique(feat_norm)) < 2){
      pvals[i] <- NA
    } else {
      pvals[i] <- tryCatch(t.test(feat_norm ~ y)$p.value, error=function(e) NA)
    }
    
    # AUC
    roc_obj <- tryCatch(roc(y, feat_norm, quiet=TRUE, levels=c("Control","Case")),
                        error=function(e) NA)
    aucs[i] <- ifelse(!is.na(roc_obj), auc(roc_obj), NA)
    
    # FoldChange and log2 FoldChange
    case_mean <- mean(feat_imp[y == "Case"], na.rm=TRUE)
    ctrl_mean <- mean(feat_imp[y == "Control"], na.rm=TRUE)
    if (is.finite(case_mean) && is.finite(ctrl_mean) && ctrl_mean > 0){
      fcs[i]  <- case_mean / ctrl_mean
      l2fc[i] <- log2(fcs[i])
      dirs[i] <- ifelse(l2fc[i] > 0, "↑", ifelse(l2fc[i] < 0, "↓", "→"))
    } else {
      fcs[i]  <- NA
      l2fc[i] <- NA
      dirs[i] <- NA
    }
  }
  
  # Criate table
  df <- data.frame(
    Feature        = names(X_norm),
    p_value        = round(pvals, 4),
    AUC            = round(aucs, 3),
    FoldChange     = round(fcs, 3),
    log2FoldChange = round(l2fc, 3),
    Direction      = dirs
  )
  
  # Sort by descending AUC (or by p_value if preferred)
  df <- df[order(-df$p_value), ]
  
  return(df)
}

# Apply to training set
features_table_all <- compute_feature_stats_all(X_train, X_train_imp, y_train)


# Save CSV
write.csv(features_table_all, file.path(Table_dir, "Selected_Features_All_with_AUC.csv"), row.names = FALSE)

cat("Full feature table saved: Feature | p_value | AUC\n")

cat("\nAll models finished. Results saved to: ", out_dir, "\n")

#==========================================================
# Model Features (p-value < 0.05 + AUC > 0.65 + Identified) 
#==========================================================

id_file <- (here("Input", "Identified_Features.csv"))

if (file.exists(id_file)) {
  
# Read identified features
id_table <- read.csv(id_file)
oout_dir <- "Results_Final"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

TableF_dir <- here("Results_Final", "Table")
dir.create(TableF_dir, recursive = TRUE, showWarnings = FALSE)

# Extract feature names
identified_features <- id_table$Feature

# Select features based on training set and intersection with identified
feats_final <- identified_features[identified_features %in% colnames(X_train)]

# Subset training and validation sets
X_train_final <- X_train[, feats_final, drop=FALSE]
X_val_final   <- X_val[, feats_final, drop=FALSE]

# Models to compare
model_list <- list(
  Logistic = "glm",
  ElasticNet = "glmnet",
  SVM_Linear = "svmLinear",
  SVM_Radial = "svmRadial",
  RandomForest = "rf"
)

results_summary <- list()

# CV Handler
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Iterate through models
for (model_name in names(model_list)) {
  cat("\n--Running: ", model_name, "...\n")
  method <- model_list[[model_name]]
  
  # Create 5-fold CV indices
  folds <- createFolds(y_train, k = 5, returnTrain = TRUE)
  all_preds <- data.frame(obs = y_train, pred_prob = NA, fold = NA)
  
  for (i in seq_along(folds)) {
    train_idx <- folds[[i]]
    test_idx <- setdiff(seq_along(y_train), train_idx)
    
    X_tr <- X_train[train_idx, feats_final, drop = FALSE]
    y_tr <- y_train[train_idx]
    X_te <- X_train[test_idx, feats_final, drop = FALSE]
    
    # Fit model
    fit <- train(
      x = X_tr,
      y = y_tr,
      method = method,
      metric = "ROC",
      trControl = ctrl,
      tuneLength = 5
    )
    
    prob <- predict(fit, newdata = X_te, type = "prob")[, "Case"]
    all_preds$pred_prob[test_idx] <- prob
    all_preds$fold[test_idx] <- i
  }
  
  # CV performance
  all_preds$pred <- factor(ifelse(all_preds$pred_prob > 0.5, "Case", "Control"),
                           levels = c("Control", "Case"))
  cm_cv <- confusionMatrix(all_preds$pred, all_preds$obs, positive = "Case")
  roc_cv <- roc(all_preds$obs, all_preds$pred_prob, levels = c("Control","Case"), positive = "Case")
  
  metrics_cv <- data.frame(
    Model = model_name,
    Dataset = "CV",
    Accuracy = round(cm_cv$overall["Accuracy"], 3),
    Sensitivity = round(cm_cv$byClass["Sensitivity"], 3),
    Specificity = round(cm_cv$byClass["Specificity"], 3),
    Precision = round(cm_cv$byClass["Precision"], 3),
    F1_Score = round(2 * (cm_cv$byClass["Precision"] * cm_cv$byClass["Sensitivity"]) /
                       (cm_cv$byClass["Precision"] + cm_cv$byClass["Sensitivity"]), 3),
    AUC = round(auc(roc_cv), 3)
  )
  
  
  # Hold-out Validation
  final_fit <- train(
    x = X_train_final,
    y = y_train,
    method = method,
    metric = "ROC",
    trControl = ctrl,
    tuneLength = 5
  )
  
  prob_val <- predict(final_fit, newdata = X_val_final, type = "prob")[, "Case"]
  pred_val <- factor(ifelse(prob_val > 0.5, "Case", "Control"), levels = c("Control", "Case"))
  
# Predictions_Val
  results_val <- data.frame(
    ID = ID_val,
    obs = y_val,
    pred = pred_val,
    prob = prob_val
  )
  write.csv(results_val, file.path(TableF_dir, paste0("Predictions_Val_", model_name, ".csv")), row.names = FALSE)
  
  cm_val <- confusionMatrix(pred_val, y_val, positive = "Case")
  roc_val <- roc(y_val, prob_val, levels = c("Control","Case"), positive = "Case")
  
  # Predictions on training set
  results_train <- data.frame(
    ID = ID_train,
    obs = y_train,
    pred = predict(final_fit, newdata = X_train_final, type = "raw"),
    prob = predict(final_fit, newdata = X_train_final, type = "prob")[, "Case"]
  )
  
  # Save training predictions
  write.csv(results_train, file.path(TableF_dir, paste0("Predictions_Train_", model_name, ".csv")), row.names = FALSE)
  
  metrics_val <- data.frame(
    Model = model_name,
    Dataset = "Validation",
    Accuracy = round(cm_val$overall["Accuracy"], 3),
    Sensitivity = round(cm_val$byClass["Sensitivity"], 3),
    Specificity = round(cm_val$byClass["Specificity"], 3),
    Precision = round(cm_val$byClass["Precision"], 3),
    F1_Score = round(2 * (cm_val$byClass["Precision"] * cm_val$byClass["Sensitivity"]) /
                       (cm_val$byClass["Precision"] + cm_val$byClass["Sensitivity"]), 3),
    AUC = round(auc(roc_val), 3)
  )
  
  # ROC Plot
  roc_dir <- here("Results_Final", "ROC")
  dir.create(roc_dir, recursive = TRUE, showWarnings = FALSE)
  
  png(file.path(roc_dir, paste0("ROC_", model_name, ".png")), width = 1600, height = 1200, res = 300)
  plot(1 - roc_cv$specificities, roc_cv$sensitivities, type = "l", col = "blue", lwd = 2,
       xlab = "False Positive Rate", ylab = "Sensitivity",
       main = paste("ROC -", model_name))
  lines(1 - roc_val$specificities, roc_val$sensitivities, col = "red", lwd = 2)
  legend("bottomright", legend = c(
    paste0("Training AUC = ", round(auc(roc_cv), 3)),
    paste0("Validation AUC = ", round(auc(roc_val), 3))
  ), col = c("blue", "red"), lwd = 2, cex = 0.9)
  dev.off()
  
  # Collect results
  results_summary[[model_name]] <- rbind(metrics_cv, metrics_val)
}

# Save results
all_results <- bind_rows(results_summary)
write.csv(all_results, file.path(TableF_dir,"Model_Performance_Summary.csv"), row.names=FALSE)

cat("\nAll models finished. Results saved to:", out_dir, "\n")
}
# Total number of original features
n_total_features <- ncol(data %>% select(-Class,-SampleID))

# Number of features after RSD filtering
n_features_rsd <- if (exists("keep_rsd")) length(keep_rsd) else NA

# Number of features after p_value <= 0.05
n_features_pval <- sum(features_table_all$p_value <= 0.05, na.rm=TRUE)

# Number of identified features
n_features_identified <- if (exists("identified_features")) length(identified_features) else NA

# Create table in "row" format
feature_summary <- data.frame(
  Metric = c("Total_Features", "Features_RSD", "Features_pval", "Features_ID"),
  Value  = c(n_total_features, n_features_rsd, n_features_pval, n_features_identified)
)

# # Save as separate CSV
write.csv(feature_summary, file.path(TableF_dir, "Feature_Summary.csv"), row.names = FALSE)

cat("Feature summary saved: Feature_Summary.csv\n")

#=====================================================
# Figures
#=====================================================
# PCA for validated alignment (QC vs Sample)

PCA_QC <- rbind(
  if (!is.null(X_QC) && nrow(X_QC)>0) X_QC else X_train[0, , drop=FALSE],
  X_train,
  X_val
)
Label_Alinhamento_Validado <- factor(c(
  if (!is.null(X_QC) && nrow(X_QC)>0) rep("QC", nrow(X_QC)) else character(0),
  rep("Sample", nrow(X_train) + nrow(X_val))
), levels=c("QC","Sample"))

stopifnot(ncol(PCA_QC) >= 2)
summary_df <- data.frame(
  any_na   = anyNA(PCA_QC),
  any_inf  = any(!is.finite(as.matrix(PCA_QC))),
  rows     = nrow(PCA_QC),
  cols     = ncol(PCA_QC)
)
print(summary_df)

pca_Alinhamento_Validado <- prcomp(PCA_QC, center=TRUE, scale.=FALSE)
df_pca_Alinhamento_Validado <- data.frame(pca_Alinhamento_Validado$x[,1:2])
names(df_pca_Alinhamento_Validado) <- c("PC1","PC2")
df_pca_Alinhamento_Validado$Label <- Label_Alinhamento_Validado
df_pca_Alinhamento_Validado$PC1 <- -df_pca_Alinhamento_Validado$PC1

pc1_var <- round(100 * pca_Alinhamento_Validado$sdev[1]^2 / sum(pca_Alinhamento_Validado$sdev^2), 2)
pc2_var <- round(100 * pca_Alinhamento_Validado$sdev[2]^2 / sum(pca_Alinhamento_Validado$sdev^2), 2)

out_dir_pca <- here("PCA","Output")

if (!dir.exists(file.path(out_dir, "pca"))) dir.create(file.path(out_dir, "pca"), recursive = TRUE)
png(file.path(out_dir, "pca" ,"Alinhamento_Validado_PCA_QC_vs_Sample.png"), width=3200, height=2400, res=600)
ggplot(df_pca_Alinhamento_Validado, aes(PC1, PC2, color=Label, fill=Label)) +
  geom_point(size=0.5, alpha=0.9) +
  stat_ellipse(data=subset(df_pca_Alinhamento_Validado, Label=="QC"),
               geom="polygon", alpha=0.4, color=NA) +
  stat_ellipse(data=subset(df_pca_Alinhamento_Validado, Label=="Sample"),
               geom="polygon", alpha=0.07, color=NA) +
  labs(title="Validated Alignment PCA (QC vs Sample)",
       x=paste0("PC1 (", pc1_var, "%)"),
       y=paste0("PC2 (", pc2_var, "%)")) +
  theme_minimal() + theme(plot.title = element_text(hjust=0.5)) +
  scale_color_manual(values=c("Sample"="#1f77b4", "QC"="#ff7f0e")) +
  scale_fill_manual(values=c("Sample"="#1f77b4", "QC"="#ff7f0e"))
dev.off()


# PCA on the training dataset (no preprocessed)
pca <- prcomp(X_train, center = TRUE, scale. = TRUE)

# Create a dataframe for plotting
df_pca <- data.frame(pca$x[, 1:2])  # PC1 and PC2
df_pca$Label <- y_train  # Corresponding classes

# Explained variance
pc1_var <- round((pca$sdev[1]^2) / sum(pca$sdev^2) * 100, 2)
pc2_var <- round((pca$sdev[2]^2) / sum(pca$sdev^2) * 100, 2)

# PCA plot
if (!dir.exists(file.path(out_dir, "pca"))) dir.create(file.path(out_dir, "pca"), recursive = TRUE)
png(file.path(out_dir,"pca","PCA_Train_Control_vs_Case (All Features).png"), width = 3200, height = 2400, res = 600)
ggplot(df_pca, aes(x = PC1, y = PC2, color = Label, fill = Label)) +
  geom_point(size = 1.5, alpha = 0.9) +  # Add points
  stat_ellipse(type = "norm", alpha = 0.1, geom = "polygon", color = NA) +  # Add confidence ellipse
  labs(title = "PCA - Training Set",
       x = paste0("PC1 (", pc1_var, "%)"),
       y = paste0("PC2 (", pc2_var, "%)")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_color_manual(values = c("Control" = "#00BFC4", "Case" = "#F8766D")) +
  scale_fill_manual(values = c("Control" = "#00BFC4", "Case" = "#F8766D"))
dev.off()

  
# PCA on the training dataset (no reprocessed)
pca <- prcomp(X_tr, center = TRUE, scale. = TRUE)

# Create a dataframe for plotting
df_pca <- data.frame(pca$x[, 1:2])  # PC1 and PC2
df_pca$Label <- y_tr # Corresponding classes

# Explained variance
pc1_var <- round((pca$sdev[1]^2) / sum(pca$sdev^2) * 100, 2)
pc2_var <- round((pca$sdev[2]^2) / sum(pca$sdev^2) * 100, 2)

# PCA plot
if (!dir.exists(file.path(out_dir, "pca"))) dir.create(file.path(out_dir, "pca"), recursive = TRUE)
png(file.path(out_dir,"pca","PCA_Train_Control_vs_Case.png"), width = 3200, height = 2400, res = 600)
ggplot(df_pca, aes(x = PC1, y = PC2, color = Label, fill = Label)) +
  geom_point(size = 1.5, alpha = 0.9) +  # Add points
  stat_ellipse(type = "norm", alpha = 0.1, geom = "polygon", color = NA) +  # Add confidence ellipse
  labs(title = "PCA - Training Set",
       x = paste0("PC1 (", pc1_var, "%)"),
       y = paste0("PC2 (", pc2_var, "%)")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_color_manual(values = c("Control" = "#00BFC4", "Case" = "#F8766D")) +
  scale_fill_manual(values = c("Control" = "#00BFC4", "Case" = "#F8766D"))
dev.off()



# 1. Convert X_train to long format
df_long <- as.data.frame(X_train_imp)
df_long$SampleID <- rownames(X_train_imp)
df_long$Group <- y_train

df_long <- df_long %>%
  tidyr::pivot_longer(
    cols = -c(SampleID, Group),
    names_to = "Feature",
    values_to = "Intensity"
  )

# 2. Join with annotations
id_file <- read_csv(here("Input","Identified_Features.csv"))

df_merged <- df_long %>%
  left_join(id_file, by = c("Feature" = "Features"))
df_merged <- df_merged %>%
  filter(!is.na(Class))

df_alluvial <- df_merged %>%
  filter(!is.na(Class)) %>%
  group_by(SampleID, Group, Class) %>%
  summarise(Abundance = sum(Intensity, na.rm = TRUE), .groups = "drop")

# Replace rare or extra classes with "Others"
df_alluvial <- df_alluvial %>%
  mutate(Class = ifelse(!(Class %in% c("Others", "Fatty Acyls", 
                                       "Glycerophospholipids", "Sphingolipids")),
                        "Others", Class))

# Create factor with desired order
df_alluvial <- df_alluvial %>%
  mutate(Class = factor(Class, levels = c(
    "Others", "Fatty Acyls", "Glycerophospholipids", "Sphingolipids"
  )))

# Group by Group and Class
df_alluvial_grouped <- df_alluvial %>%
  group_by(Group, Class) %>%
  summarise(Abundance = sum(Abundance), .groups = "drop")

# 4a. Create color column for stratum (Group or Class)
df_alluvial_grouped <- df_alluvial_grouped %>%
  mutate(StratumFill = ifelse(
    Class %in% c("Others", "Fatty Acyls", "Glycerophospholipids", "Sphingolipids"),
    as.character(Class),  # Class axis
    as.character(Group)   # Group axis
  ))

total_abundance <- sum(df_alluvial_grouped$Abundance)

df_alluvial_grouped <- df_alluvial_grouped %>%
  mutate(AbundancePct = Abundance / total_abundance * 100)

# --- Define colors ---

class_colors <- c(
  "Others"               = "pink",
  "Fatty Acyls"          = "#83bb3f",
  "Glycerophospholipids" = "#ADD8E6",
  "Sphingolipids"        = "#FFE87C"
)

# --- Alluvial plot ---
ggplot(df_alluvial_grouped,
       aes(axis1 = Group, axis2 = Class, y = AbundancePct)) +
  
  # flows colored by classes
  geom_alluvium(aes(fill = Class), width = 1/12, alpha = 0.85) +
  
  # strata colored by Group or Class
  geom_stratum(aes(fill = StratumFill), width = 1/12, color = "black") +
  
  # labels only on Group axis
  geom_text(stat = "stratum",
            aes(label = ifelse(..x.. == 1, as.character(stratum), "")),
            size = 7, color = "black",
            angle = 90, vjust = 0.5, hjust = 0.5) +
  
  scale_x_discrete(limits = c("Group", "Class"), expand = c(0.02, 0))+
  scale_y_continuous(
    expand = expansion(mult = c(0.05, 0.1)),
    labels = function(x) paste0(x, "%")   # display as percentage
  ) +
  
  scale_fill_manual(
    values = c(class_colors),
    na.value = "gray80",
    breaks = c(names(class_colors))
  ) +
  
  labs(y = "Percentage", x = "") +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),  
    panel.grid.minor = element_blank(),  
    axis.text.x  = element_text(size = 14),
    axis.text.y  = element_text(size = 12),
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    legend.text  = element_text(size = 12),
    legend.title = element_text(size = 14)
  )

# Save
dir.create(file.path("Results_Final", "Alluvial"), recursive = TRUE, showWarnings = FALSE)
ggsave(filename = file.path("Results_Final", "Alluvial", "alluvial_plot.png"),
       width = 10, height = 7, units = "in", dpi = 300)

# Read identified features
id_file <- (here("Input", "Identified_Features.csv"))
identified_features <- read.csv(id_file, stringsAsFactors = FALSE)

# Normalize column names (remove spaces, convert to lowercase)
colnames(identified_features) <- str_trim(colnames(identified_features))
colnames(identified_features) <- str_to_lower(colnames(identified_features))

# Check column names (just to verify)
print(colnames(identified_features))
# Should appear: "features" "name" "class"

# Filter only features present in X_train
X_train_sel <- X_train[, colnames(X_train) %in% identified_features$features, drop = FALSE]

# Create dataframe features x name
feature_names <- identified_features %>%
  filter(features %in% colnames(X_train_sel)) %>%
  select(features, name)

# Transform into long format and add Name column
X_train_long <- X_train_sel %>%
  mutate(Class = y_train) %>%
  pivot_longer(
    cols = -Class,
    names_to = "Feature",
    values_to = "Value"
  ) %>%
  left_join(feature_names, by = c("Feature" = "features"))


nm <- ggplot(X_train_long, aes(x = name, y = Value, fill = Class)) +
  geom_boxplot(
    width = 0.6,
    position = position_dodge(width = 0.8),
    outlier.shape = NA,
    fatten = 0.5)+   # <--- reduce median line thickness
  scale_fill_manual(
    values = c("Case" = "#F8766D", "Control" = "#00BFC4"),
    breaks = c("Case", "Control"),
    labels = c("SCLUs", "SCD")   # <--- change legend text
  ) +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  ) +
  labs(
    title = "Boxplot - Compounds",
    x = "Metabolite",
    y = "Intensity",
    fill = "Class"
  )
# Save as PDF (good quality for paper)
dir.create(here("Results_Final", "BoxPlot"), recursive = TRUE, showWarnings = FALSE)

ggsave(here("Results_Final", "BoxPlot","BoxPlot_ALL.png"), 
       plot = nm, width = 10, height = 6)

sphingo_feats <- identified_features %>%
  filter(subclass == "Sphingomyelins") %>%
  pull("features")

# Check if these features exist in processed data
sphingo_feats <- sphingo_feats[sphingo_feats %in% colnames(X_train)]

if (length(sphingo_feats) > 0) {
  # Combine data (train + validation) for plotting
  df_plot <- rbind(
    data.frame(ID = ID_train, Class = y_train, X_train[, sphingo_feats, drop=FALSE]),
    data.frame(ID = ID_val,   Class = y_val,   X_val[,   sphingo_feats, drop=FALSE])
  )
  
  df_long <- df_plot %>%
    pivot_longer(
      cols = all_of(sphingo_feats),
      names_to = "Feature",
      values_to = "Intensity"
    ) %>%
    left_join(
      identified_features %>% select(features, name, `p.value`),
      by = c("Feature" = "features")
    ) %>%
    mutate(
      # create label
      facet_label = paste0(name,  "<br><i>p-value = ", signif(p.value, 3)),
      facet_label = str_replace_all(facet_label, "\n", "<br>"),
      Class = factor(Class,
                     levels = c("Control", "Case"),
                     labels = c("SCD", "SCLUs")))
  
  p <- ggplot(df_long, aes(x = "", y = Intensity, fill = Class)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.7, fatten = 0.5) +
    facet_wrap(~facet_label, scales = "free_y") +
    theme_bw(base_size = 12) +
    theme(strip.text = element_markdown())+
    labs(title="Boxplots - Sphingomyelins",
         y="Intensity",
         x="Group",
         fill="Class") +
    
    scale_fill_manual(values=c("SCD"="#00BFC4", "SCLUs"="#F8766D"))+
    theme(
      plot.title = element_text(hjust = 0.5, vjust = 1.5),
      legend.position = c(0.78, 0.05),   # <<< relative position inside panel
      legend.justification = c("left", "bottom"), 
      legend.background = element_rect(fill = scales::alpha("white", 0.7), color = NA),
      legend.text = element_text(size = 20),   # <<< legend item size
      legend.title = element_text(size = 21)   # <<< legend title size
    )
  
  # Save picture
  dir.create(here("Results_Final", "BoxPlot"), recursive = TRUE, showWarnings = FALSE)
  
  ggsave(here("Results_Final", "BoxPlot","BoxPlot_Sphingomyelins.png"), 
         plot = p, width = 10, height = 6)
  
  cat("Boxplot generated: Results_Final/BoxPlot/BoxPlot_Sphingomyelins.pdf\n")
} else {
  cat("No feature SubClass = 'Sphingomyelins' found in the data.\n")
}
