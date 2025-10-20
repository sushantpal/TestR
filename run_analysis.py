import csv
import math
from pathlib import Path

# Embedded mtcars dataset
columns = [
    "mpg", "cyl", "disp", "hp", "drat", "wt", "qsec", "vs", "am", "gear", "carb"
]
data_rows = [
    (21.0, 6, 160.0, 110, 3.90, 2.620, 16.46, 0, 1, 4, 4),
    (21.0, 6, 160.0, 110, 3.90, 2.875, 17.02, 0, 1, 4, 4),
    (22.8, 4, 108.0, 93, 3.85, 2.320, 18.61, 1, 1, 4, 1),
    (21.4, 6, 258.0, 110, 3.08, 3.215, 19.44, 1, 0, 3, 1),
    (18.7, 8, 360.0, 175, 3.15, 3.440, 17.02, 0, 0, 3, 2),
    (18.1, 6, 225.0, 105, 2.76, 3.460, 20.22, 1, 0, 3, 1),
    (14.3, 8, 360.0, 245, 3.21, 3.570, 15.84, 0, 0, 3, 4),
    (24.4, 4, 146.7, 62, 3.69, 3.190, 20.00, 1, 0, 4, 2),
    (22.8, 4, 140.8, 95, 3.92, 3.150, 22.90, 1, 0, 4, 2),
    (19.2, 6, 167.6, 123, 3.92, 3.440, 18.30, 1, 0, 4, 4),
    (17.8, 6, 167.6, 123, 3.92, 3.440, 18.90, 1, 0, 4, 4),
    (16.4, 8, 275.8, 180, 3.07, 4.070, 17.40, 0, 0, 3, 3),
    (17.3, 8, 275.8, 180, 3.07, 3.730, 17.60, 0, 0, 3, 3),
    (15.2, 8, 275.8, 180, 3.07, 3.780, 18.00, 0, 0, 3, 3),
    (10.4, 8, 472.0, 205, 2.93, 5.250, 17.98, 0, 0, 3, 4),
    (10.4, 8, 460.0, 215, 3.00, 5.424, 17.82, 0, 0, 3, 4),
    (14.7, 8, 440.0, 230, 3.23, 5.345, 17.42, 0, 0, 3, 4),
    (32.4, 4, 78.7, 66, 4.08, 2.200, 19.47, 1, 1, 4, 1),
    (30.4, 4, 75.7, 52, 4.93, 1.615, 18.52, 1, 1, 4, 2),
    (33.9, 4, 71.1, 65, 4.22, 1.835, 19.90, 1, 1, 4, 1),
    (21.5, 4, 120.1, 97, 3.70, 2.465, 20.01, 1, 0, 3, 1),
    (15.5, 8, 318.0, 150, 2.76, 3.520, 16.87, 0, 0, 3, 2),
    (15.2, 8, 304.0, 150, 3.15, 3.435, 17.30, 0, 0, 3, 2),
    (13.3, 8, 350.0, 245, 3.73, 3.840, 15.41, 0, 0, 3, 4),
    (19.2, 8, 400.0, 175, 3.08, 3.845, 17.05, 0, 0, 3, 2),
    (27.3, 4, 79.0, 66, 4.08, 1.935, 18.90, 1, 1, 4, 1),
    (26.0, 4, 120.3, 91, 4.43, 2.140, 16.70, 0, 1, 5, 2),
    (30.4, 4, 95.1, 113, 3.77, 1.513, 16.90, 1, 1, 5, 2),
    (15.8, 8, 351.0, 264, 4.22, 3.170, 14.50, 0, 1, 5, 4),
    (19.7, 6, 145.0, 175, 3.62, 2.770, 15.50, 0, 1, 5, 6),
    (15.0, 8, 301.0, 335, 3.54, 3.570, 14.60, 0, 1, 5, 8),
    (21.4, 4, 121.0, 109, 4.11, 2.780, 18.60, 1, 1, 4, 2),
]

with open("config/train_indices.csv") as f:
    train_indices = [int(x.strip()) for x in f.read().split(',') if x.strip()]

# Convert to zero-based indices
train_indices = [i - 1 for i in train_indices]
all_indices = list(range(len(data_rows)))
test_indices = [i for i in all_indices if i not in train_indices]

# Matrix helpers

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def matmul(a, b):
    result = [[0.0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            result[i][j] = sum(a[i][k] * b[k][j] for k in range(len(b)))
    return result


def matvec(a, v):
    return [sum(a[i][k] * v[k] for k in range(len(v))) for i in range(len(a))]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def invert(matrix):
    n = len(matrix)
    aug = [row[:] + identity_row[:] for row, identity_row in zip(matrix, identity(n))]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("Matrix is singular")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        aug[col] = [val / pivot for val in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [aug[row][c] - factor * aug[col][c] for c in range(2 * n)]
    return [row[n:] for row in aug]


# Prepare design matrices

def build_design_matrix(indices):
    X = []
    y = []
    for idx in indices:
        row = data_rows[idx]
        y.append(row[0])
        X.append([1.0] + list(row[1:]))
    return X, y


X_train, y_train = build_design_matrix(train_indices)
X_test, y_test = build_design_matrix(test_indices)

# Normal equation: beta = (X^T X)^{-1} X^T y
Xt = transpose(X_train)
XtX = matmul(Xt, X_train)
XtX_inv = invert(XtX)
Xty = matvec(Xt, y_train)

beta = matvec(XtX_inv, Xty)

# Predictions
def predict_row(row):
    return beta[0] + sum(beta[i + 1] * row[i + 1] for i in range(len(row) - 1))

predictions = [predict_row(row) for row in X_test]

# Metrics
rmse = math.sqrt(sum((pred - actual) ** 2 for pred, actual in zip(predictions, y_test)) / len(y_test))
mae = sum(abs(pred - actual) for pred, actual in zip(predictions, y_test)) / len(y_test)
mean_actual = sum(y_test) / len(y_test)
ss_total = sum((actual - mean_actual) ** 2 for actual in y_test)
ss_res = sum((actual - pred) ** 2 for pred, actual in zip(predictions, y_test))
r_squared = 1 - ss_res / ss_total if ss_total != 0 else float("nan")

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

metrics_path = output_dir / "metrics.csv"
with metrics_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["RMSE", f"{rmse:.6f}"])
    writer.writerow(["MAE", f"{mae:.6f}"])
    writer.writerow(["R_Squared", f"{r_squared:.6f}"])

kpi_path = output_dir / "kpi_summary.csv"
with kpi_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["KPI", "Value"])
    writer.writerow(["Average MPG", f"{sum(row[0] for row in data_rows) / len(data_rows):.6f}"])
    writer.writerow(["Average Horsepower", f"{sum(row[3] for row in data_rows) / len(data_rows):.6f}"])
    writer.writerow(["Average Weight", f"{sum(row[5] for row in data_rows) / len(data_rows):.6f}"])
    writer.writerow(["Average Quarter Mile Time", f"{sum(row[6] for row in data_rows) / len(data_rows):.6f}"])

predictions_path = output_dir / "predictions.csv"
with predictions_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Car_Index", "Actual_MPG", "Predicted_MPG"])
    for idx, actual, pred in zip(test_indices, y_test, predictions):
        writer.writerow([idx + 1, f"{actual:.6f}", f"{pred:.6f}"])

# Model summary placeholder
summary_path = output_dir / "model_summary.txt"
with summary_path.open("w") as f:
    f.write("Linear regression coefficients (beta):\n")
    for name, coef in zip(["Intercept"] + columns[1:], beta):
        f.write(f"{name}: {coef:.6f}\n")
    f.write("\nTrain observations: {}\n".format(len(train_indices)))
    f.write("Test observations: {}\n".format(len(test_indices)))
    f.write(f"RMSE: {rmse:.6f}\n")
    f.write(f"MAE: {mae:.6f}\n")
    f.write(f"R-squared: {r_squared:.6f}\n")

# Generate simple SVG scatter plot
svg_path = output_dir / "actual_vs_predicted.svg"
width, height = 700, 500
margin = 60

actual_values = y_test
pred_values = predictions

min_actual = min(actual_values)
max_actual = max(actual_values)
min_pred = min(pred_values)
max_pred = max(pred_values)

def scale(value, min_val, max_val, length):
    if max_val == min_val:
        return margin
    return margin + (value - min_val) / (max_val - min_val) * length

content = [
    f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
    "<style>text { font-family: Arial, sans-serif; font-size: 14px; }</style>",
    f"<rect width='{width}' height='{height}' fill='white' stroke='none' />",
]

plot_width = width - 2 * margin
plot_height = height - 2 * margin

# Axes
content.append(f"<line x1='{margin}' y1='{height - margin}' x2='{margin + plot_width}' y2='{height - margin}' stroke='black' stroke-width='2' />")
content.append(f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='black' stroke-width='2' />")

# Axis labels
content.append(f"<text x='{width/2}' y='{height - 15}' text-anchor='middle'>Actual MPG</text>")
content.append(f"<text x='15' y='{height/2}' transform='rotate(-90 15,{height/2})' text-anchor='middle'>Predicted MPG</text>")
content.append(f"<text x='{width/2}' y='30' text-anchor='middle'>Actual vs Predicted MPG</text>")
content.append(f"<text x='{width/2}' y='50' text-anchor='middle'>Linear regression using mtcars dataset</text>")

# Diagonal line
x1 = scale(min_actual, min_actual, max_actual, plot_width)
y1 = height - scale(min_actual, min_pred, max_pred, plot_height)
x2 = scale(max_actual, min_actual, max_actual, plot_width)
y2 = height - scale(max_actual, min_pred, max_pred, plot_height)
content.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#E74C3C' stroke-dasharray='5,5' stroke-width='2' />")

# Points
for actual, pred in zip(actual_values, pred_values):
    x = scale(actual, min_actual, max_actual, plot_width)
    y = height - scale(pred, min_pred, max_pred, plot_height)
    content.append(f"<circle cx='{x}' cy='{y}' r='5' fill='#2C3E50' />")

content.append("</svg>")

with svg_path.open("w") as f:
    f.write("\n".join(content))

print("RMSE", rmse)
print("MAE", mae)
print("R-squared", r_squared)
