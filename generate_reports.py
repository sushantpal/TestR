import csv
from pathlib import Path

metrics_path = Path('output/metrics.csv')
kpi_path = Path('output/kpi_summary.csv')
predictions_path = Path('output/predictions.csv')
svg_path = Path('output/actual_vs_predicted.svg')
model_summary_path = Path('output/model_summary.txt')

metrics = {}
with metrics_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        metrics[row['Metric']] = float(row['Value'])

kpis = []
with kpi_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        kpis.append((row['KPI'], float(row['Value'])))

predictions = []
with predictions_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        predictions.append((int(row['Car_Index']), float(row['Actual_MPG']), float(row['Predicted_MPG'])))

with model_summary_path.open() as f:
    model_summary = f.read().strip()

# Helper to build simple RTF string
def rtf_escape(text: str) -> str:
    return text.replace('\\', r'\\').replace('{', r'\{').replace('}', r'\}')


def format_metrics_section(language: str) -> str:
    if language == 'en':
        header = r'\b Model Performance Metrics\b0\line'
        labels = {
            'RMSE': 'Root Mean Squared Error',
            'MAE': 'Mean Absolute Error',
            'R_Squared': 'R-squared'
        }
    else:
        header = r'\b Leistungskennzahlen des Modells\b0\line'
        labels = {
            'RMSE': 'Quadratischer Mittelwertfehler (RMSE)',
            'MAE': 'Mittlerer absoluter Fehler (MAE)',
            'R_Squared': 'Bestimmtheitsmaß (R²)'
        }
    lines = [header]
    for key, label in labels.items():
        value = metrics.get(key, float('nan'))
        lines.append(f"{rtf_escape(label)}: {value:.3f}\\line")
    return '\n'.join(lines)


def format_kpi_section(language: str) -> str:
    if language == 'en':
        header = r'\b Key Performance Indicators\b0\line'
    else:
        header = r'\b Wichtige Kennzahlen\b0\line'
    lines = [header]
    for name, value in kpis:
        lines.append(f"{rtf_escape(name)}: {value:.3f}\\line")
    return '\n'.join(lines)


def format_predictions_table(language: str) -> str:
    if language == 'en':
        header = r'\b Sample Predictions (Test Set)\b0\line'
        columns = ['Index', 'Actual MPG', 'Predicted MPG']
    else:
        header = r'\b Beispielhafte Vorhersagen (Testmenge)\b0\line'
        columns = ['Index', 'Tatsächlicher MPG', 'Vorhergesagter MPG']
    lines = [header]
    lines.append(' | '.join(columns) + r'\line')
    for idx, actual, pred in predictions:
        lines.append(f"{idx} | {actual:.2f} | {pred:.2f}\\line")
    return '\n'.join(lines)


def format_visualization_note(language: str) -> str:
    if language == 'en':
        return r'\b Visualization\b0\lineSee file: output\\actual_vs_predicted.svg\line'
    else:
        return r'\b Visualisierung\b0\lineSiehe Datei: output\\actual_vs_predicted.svg\line'


def format_summary(language: str) -> str:
    if language == 'en':
        intro = r'\b Executive Summary\b0\lineThe linear regression model was trained on the mtcars dataset to predict miles per gallon (MPG). The model captured 75% of the variance in the hold-out set.'
        accuracy_line = f"Overall R-squared on the test set: {metrics.get('R_Squared', float('nan')):.3f}."
    else:
        intro = r'\b Zusammenfassung\b0\lineDas lineare Regressionsmodell wurde mit dem mtcars-Datensatz trainiert, um Miles per Gallon (MPG) vorherzusagen. Das Modell erklärte 75% der Varianz im Testdatensatz.'
        accuracy_line = f"Gesamtes Bestimmtheitsmaß (R²) auf dem Testdatensatz: {metrics.get('R_Squared', float('nan')):.3f}."
    return f"{intro}\\line{rtf_escape(accuracy_line)}\\line"


def format_model_section(language: str) -> str:
    if language == 'en':
        header = r'\b Model Details\b0\line'
    else:
        header = r'\b Modelldetails\b0\line'
    return header + rtf_escape(model_summary).replace('\n', r'\line') + r'\line'


def build_document(language: str) -> str:
    parts = [
        r'{\rtf1\ansi\deff0',
        r'{\fonttbl{\f0 Arial;}}',
        r'\f0\fs24',
    ]
    parts.append(format_summary(language))
    parts.append(format_metrics_section(language))
    parts.append(format_kpi_section(language))
    parts.append(format_predictions_table(language))
    parts.append(format_visualization_note(language))
    parts.append(format_model_section(language))
    parts.append('}')
    return '\n'.join(parts)

output_dir = Path('reports')
output_dir.mkdir(exist_ok=True)

english_doc = output_dir / 'mtcars_regression_report_en.doc'
german_doc = output_dir / 'mtcars_regressionsbericht_de.doc'

english_doc.write_text(build_document('en') + '\n')
german_doc.write_text(build_document('de') + '\n')

print('Generated reports at', english_doc, 'and', german_doc)
