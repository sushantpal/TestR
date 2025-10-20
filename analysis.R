# Analysis script for hama_cycle_data dataset
# Performs linear regression and outputs KPIs, metrics, and visualization

# Ensure required packages are available
required_packages <- c("ggplot2", "svglite")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

suppressPackageStartupMessages(library(ggplot2))

set.seed(123)

data_path <- "hama_cycle_data.csv"

if (!file.exists(data_path)) {
  stop(sprintf("Input data file not found at '%s'", data_path))
}

data <- read.csv(data_path, stringsAsFactors = FALSE)

# Train/test split (70/30) with deterministic indices if config file exists
config_dir <- "config"
if (!dir.exists(config_dir)) {
  dir.create(config_dir, recursive = TRUE)
}
indices_path <- file.path(config_dir, "train_indices.csv")

if (file.exists(indices_path)) {
  train_idx <- scan(indices_path, what = integer(), sep = ",", quiet = TRUE)
} else {
  train_idx <- sample(seq_len(nrow(data)), size = floor(0.7 * nrow(data)))
  write(train_idx, file = indices_path, sep = ",")
}

train_data <- data[train_idx, ]
test_data <- data[-train_idx, ]

model <- lm(mpg ~ ., data = train_data)

predictions <- predict(model, newdata = test_data)
actuals <- test_data$mpg

rmse <- sqrt(mean((predictions - actuals)^2))
mae <- mean(abs(predictions - actuals))
ss_total <- sum((actuals - mean(actuals))^2)
ss_res <- sum((actuals - predictions)^2)
r_squared <- 1 - (ss_res / ss_total)

output_dir <- "output"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

metrics <- data.frame(
  Metric = c("RMSE", "MAE", "R_Squared"),
  Value = c(rmse, mae, r_squared)
)

write.csv(metrics, file = file.path(output_dir, "metrics.csv"), row.names = FALSE)

kpi_summary <- data.frame(
  KPI = c("Average MPG", "Average Horsepower", "Average Weight", "Average Quarter Mile Time"),
  Value = c(mean(data$mpg), mean(data$hp), mean(data$wt), mean(data$qsec))
)

write.csv(kpi_summary, file = file.path(output_dir, "kpi_summary.csv"), row.names = FALSE)

model_summary <- summary(model)

capture.output(model_summary, file = file.path(output_dir, "model_summary.txt"))

plot_data <- data.frame(Actual = actuals, Predicted = predictions)

plot <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = "#2C3E50", size = 3) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#E74C3C") +
  theme_minimal() +
  labs(
    title = "Actual vs Predicted MPG",
    subtitle = "Linear regression using hama_cycle_data dataset",
    x = "Actual MPG",
    y = "Predicted MPG"
  )

ggsave(filename = file.path(output_dir, "actual_vs_predicted.svg"), plot = plot, width = 7, height = 5, dpi = 300, device = "svglite")

# Save predictions vs actuals for reporting
pred_table <- data.frame(Car = rownames(test_data), Actual_MPG = actuals, Predicted_MPG = predictions)
write.csv(pred_table, file = file.path(output_dir, "predictions.csv"), row.names = FALSE)

# Print metrics to console for quick inspection
print(metrics)
print(kpi_summary)

