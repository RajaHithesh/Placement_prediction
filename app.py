"""Placement Prediction System - Flask front end.

Serves the student-placement analysis, charts and trained models produced by the
project's Module 1-5 scripts. This app only READS project files: the raw dataset
and every model artefact are never modified.
"""

import os
import joblib
import re

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory, abort

# ================================================================
# PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_FILE = "placement_predict_50K_Raw.csv"
DATASET_PATH = os.path.join(DATASET_DIR, DATASET_FILE)

app = Flask(__name__)

# ================================================================
# NAVIGATION (drives the sidebar rail and the ⌘K command palette)
# ================================================================

NAV_SECTIONS = [
    {
        "label": "Overview",
        "items": [
            {"id": "home", "label": "Dashboard", "icon": "◈", "endpoint": "home",
             "desc": "signals, KPIs and the live readiness console"},
            {"id": "about", "label": "About the project", "icon": "◇", "endpoint": "about",
             "desc": "goal, workflow and technology"},
        ],
    },
    {
        "label": "Data",
        "items": [
            {"id": "dataset", "label": "Dataset", "icon": "▤", "endpoint": "dataset",
             "desc": "50,000 student records and the column dictionary"},
            {"id": "preprocessing", "label": "Preprocessing", "icon": "⚙", "endpoint": "preprocessing",
             "desc": "cleaning, encoding, scaling and feature selection"},
            {"id": "visualization", "label": "Visualization", "icon": "◔", "endpoint": "visualization",
             "desc": "every exploratory chart, grouped and filterable"},
        ],
    },
    {
        "label": "Intelligence",
        "items": [
            {"id": "models", "label": "ML Models", "icon": "⬡", "endpoint": "models",
             "desc": "20 models compared, ranked and explained"},
            {"id": "prediction", "label": "Prediction Lab", "icon": "◎", "endpoint": "prediction",
             "tag": "live", "desc": "score a student profile with the trained model"},
        ],
    },
    {
        "label": "Output",
        "items": [
            {"id": "reports", "label": "Reports", "icon": "▦", "endpoint": "reports",
             "desc": "every generated chart, metric and file"},
        ],
    },
]


@app.context_processor
def inject_nav():
    """Give every template the rail structure and the current page."""
    current = None
    for section in NAV_SECTIONS:
        for item in section["items"]:
            if request.endpoint == item["endpoint"]:
                current = {**item, "section": section["label"]}
    return {
        "nav_sections": NAV_SECTIONS,
        "current_nav": current,
        "total_models": TOTAL_MODELS,
    }


# ================================================================
# DATASET FACTS (verified from the raw file, never guessed)
# ================================================================

TOTAL_RECORDS = 50000
TOTAL_COLUMNS = 32
PLACED_COUNT = 32856
NOT_PLACED_COUNT = 17144
PLACED_PCT = 65.712
NOT_PLACED_PCT = 34.288
MISSING_CELLS = 19976
MISSING_COLUMNS = 5
SELECTED_FEATURES = 12        # kept by Lasso out of 31 input features
TOTAL_MODELS = 21             # distinct models + methods trained - see OVERVIEW below
BEST_CLUSTERS = 3             # agglomerative and divisive agree on K = 3
RF_ACCURACY = 99.82           # % — Random Forest, PlacementStatus
DT_ACCURACY = 99.77           # % — Decision Tree, PlacementStatus

# A model that always answers "placed" — the lazy baseline to beat.
BASELINE_ACCURACY = PLACED_COUNT / TOTAL_RECORDS
BASELINE_F1 = 2 * BASELINE_ACCURACY / (BASELINE_ACCURACY + 1)

# ================================================================
# TRAINED MODEL (Module 3 pipeline, trained on the raw dataset)
# ================================================================

# The DEPLOYMENT model. Module 3's tree scores 99.77% but leans on the
# Salary Package column, which is only known AFTER a placement. This is the
# honest replacement trained in src/Deployment_Model_LeakageFree_M6.py:
# same features minus StudentID / Salary Package / IsAnomaly.
MODEL_FILE = os.path.join(
    OUTPUTS_DIR, "Deployment_Model_M6_Outputs", "model", "placement_clean_model.pkl"
)
MODEL_LABEL = "Random Forest (leakage-free, depth 10) · full sklearn pipeline"

# Metrics of the deployment model, read from
# outputs/Deployment_Model_M6_Outputs/metrics/leakage_free_metrics.csv
DEPLOY_METRICS = {
    "model": "Random Forest (leakage-free)",
    "accuracy": 0.9152,
    "precision": 0.9179,
    "recall": 0.9565,
    "f1": 0.9368,
    "roc_auc": 0.9766,
    "cv_f1": 0.9407,
    "features": 28,
    "dropped": ["StudentID", "Salary Package", "IsAnomaly"],
    "source": "src/Deployment_Model_LeakageFree_M6.py",
}

LEAKAGE_NOTE = (
    "Module 3's models report ~99.8% because the training file contains "
    "Salary Package, a column only known AFTER a student is placed. That is target "
    "leakage: the score is real, but it cannot forecast anything. The deployed model "
    "drops StudentID, Salary Package and IsAnomaly and scores an honest 91.5% — "
    "which is why the Prediction pages serve it instead."
)

_model_cache = {}


def load_model():
    """Load the deployed pipeline once and keep it in memory.

    joblib is used rather than plain pickle: sklearn models are written with
    joblib's numpy pickler, and reading them with the stdlib pickle raises
    "STACK_GLOBAL requires str".
    """
    if "deployment" not in _model_cache:
        _model_cache["deployment"] = joblib.load(MODEL_FILE)
    return _model_cache["deployment"]


# ================================================================
# PREDICTION FORM
#
# The pipeline was trained on the raw 31 feature columns. Numeric inputs are
# imputed with the training median and categorical ones with the most frequent
# value, so any field left blank is filled by the pipeline itself.
# StudentID, Salary Package and IsAnomaly are never asked for; they are passed
# as missing values on purpose (see the note on the Prediction page).
# ================================================================

CATEGORICAL_COLUMNS = [
    "Gender",
    "City",
    "CollegeTier",
    "Stream",
    "Specialisation",
    "Hostel",
    "HistoryOfBacklogs",
    "CGPA_Tier",
]

CATEGORICAL_PREFERENCE = {
    "Gender": ["Male", "Female"],
    "City": [
        "Pune", "Mumbai", "Bangalore", "Hyderabad", "Chennai",
        "Delhi", "Kolkata", "Ahmedabad", "Jaipur", "Kochi",
    ],
    "CollegeTier": ["Tier1", "Tier2", "Tier3"],
    "Stream": ["CS", "IT", "ECE", "EE", "Mechanical", "Civil"],
    "Specialisation": ["AI", "DataScience", "Embedded", "Networking"],
    "Hostel": ["Yes", "No"],
    "HistoryOfBacklogs": ["No", "Yes"],
    "CGPA_Tier": ["Low", "Mid", "High"],
}

CATEGORICAL_LABELS = {
    "Gender": "Gender",
    "City": "City",
    "CollegeTier": "College tier",
    "Stream": "Stream",
    "Specialisation": "Specialisation",
    "Hostel": "Hostel",
    "HistoryOfBacklogs": "History of backlogs",
    "CGPA_Tier": "CGPA tier",
}

CATEGORICAL_HINTS = {
    "CollegeTier": "Tier1 is the strongest college band",
    "HistoryOfBacklogs": "Yes means the student had backlogs earlier",
    "CGPA_Tier": "Low / Mid / High CGPA band stored in the dataset",
}


def _read_categorical_options():
    """Read the real category values from the raw dataset (with a tidy order)."""
    try:
        frame = pd.read_csv(DATASET_PATH, usecols=CATEGORICAL_COLUMNS)
    except Exception:
        return {col: list(CATEGORICAL_PREFERENCE.get(col, [])) for col in CATEGORICAL_COLUMNS}

    options = {}
    for col in CATEGORICAL_COLUMNS:
        values = sorted({str(v) for v in frame[col].dropna().unique()})
        preferred = CATEGORICAL_PREFERENCE.get(col, [])
        ordered = [v for v in preferred if v in values]
        ordered += [v for v in values if v not in ordered]
        options[col] = ordered
    return options


CATEGORICAL_OPTIONS = _read_categorical_options()


def _num(name, label, step, value, hint, low=0, high=100):
    return {
        "name": name,
        "label": label,
        "type": "number",
        "min": low,
        "max": high,
        "step": step,
        "value": value,
        "placeholder": "median",
        "hint": hint,
    }


FIELD_GROUPS = [
    {
        "icon": "🎓",
        "tone": "tint-mint",
        "title": "Academics",
        "blurb": "Grades and attendance — the strongest academic signals in the data.",
        "fields": [
            _num("CGPA", "CGPA", "0.01", "7.25", "0 – 10 · data median 7.25", 0, 10),
            _num("AttendancePercent", "Attendance %", "0.1", "77.2", "50 – 100 · data median 77.2"),
            _num("SGPA_Sem1", "SGPA · Semester 1", "0.01", "7.19", "data median 7.19", 0, 10),
            _num("SGPA_Sem2", "SGPA · Semester 2", "0.01", "7.21", "data median 7.21", 0, 10),
            _num("SGPA_Sem3", "SGPA · Semester 3", "0.01", "7.22", "data median 7.22", 0, 10),
            _num("SGPA_Sem4", "SGPA · Semester 4", "0.01", "7.24", "data median 7.24", 0, 10),
            _num("SGPA_Sem5", "SGPA · Semester 5", "0.01", "7.26", "data median 7.26", 0, 10),
            _num("SGPA_Sem6", "SGPA · Semester 6", "0.01", "7.28", "data median 7.28", 0, 10),
            _num("SGPA_Sem7", "SGPA · Semester 7", "0.01", "7.30", "data median 7.30", 0, 10),
            _num("SGPA_Sem8", "SGPA · Semester 8", "0.01", "7.31", "data median 7.31", 0, 10),
        ],
    },
    {
        "icon": "🧠",
        "tone": "tint-peach",
        "title": "Skills, tests & activities",
        "blurb": "Written test scores, soft skills and everything the student did outside class.",
        "fields": [
            _num("AptitudeTestScore", "Aptitude test score", "0.1", "69.9", "30 – 100 · data median 69.9"),
            _num("SoftSkillsRating", "Soft-skills rating", "0.1", "3.2", "1 – 5 · data median 3.2", 0, 5),
            _num("CodingTestScore", "Coding test score", "0.1", "57.7", "0 – 100 · data median 57.7"),
            _num("MockInterviewScore", "Mock interview score", "0.1", "54.7", "16 – 100 · data median 54.7"),
            _num("Internships", "Internships", "1", "1", "0 – 3 · data median 1", 0, 10),
            _num("Projects", "Projects", "1", "2", "0 – 5 · data median 2", 0, 10),
            _num("Workshops", "Workshops attended", "1", "2", "0 – 4 · data median 2", 0, 10),
            _num("Certifications", "Certifications", "1", "2", "0 – 5 · data median 2", 0, 10),
            _num("Publications", "Publications", "1", "1", "0 – 3 · data median 1", 0, 10),
            _num("ExtraCurricular", "Extra-curricular", "1", "1", "0 or 1 · data median 1", 0, 1),
        ],
    },
    {
        "icon": "🏫",
        "tone": "tint-lilac",
        "title": "Background",
        "blurb": "College, course and personal background — the categorical columns.",
        "fields": [
            {
                "name": col,
                "label": CATEGORICAL_LABELS[col],
                "type": "select",
                "options": CATEGORICAL_OPTIONS.get(col, []),
                "hint": CATEGORICAL_HINTS.get(col, ""),
            }
            for col in CATEGORICAL_COLUMNS
        ],
    },
]


def default_form_values():
    """Every field's starting value (data medians and real category values)."""
    values = {}
    for group in FIELD_GROUPS:
        for field in group["fields"]:
            if field["type"] == "select":
                options = field["options"]
                values[field["name"]] = options[0] if options else ""
            else:
                values[field["name"]] = field["value"]
    return values


# ================================================================
# SERVE CHARTS AND FILES FROM THE OUTPUTS FOLDER
# ================================================================


@app.route("/outputs/<path:filename>")
def outputs_file(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


# ================================================================
# PAGES
# ================================================================


@app.route("/")
def home():
    """Placement intelligence dashboard.

    Everything here comes from the real project files: the correlation matrix,
    the saved metric files and the trained model behind /api/predict.
    """
    by_name = {m["name"]: m for m in _collect_modules()}
    total_charts = sum(
        by_name[folder]["n_images"] for folder in M1_FOLDERS if folder in by_name
    )

    leaderboard = [row for row in CLASSIFICATION_COMPARISON if not row.get("ref")]

    pipeline = [
        {"step": "M1", "title": "Load & explore", "desc": "50,000 records, missing values mapped",
         "endpoint": "visualization", "icon": "◔"},
        {"step": "M2", "title": "Clean & encode", "desc": "impute, encode, scale, select features",
         "endpoint": "preprocessing", "icon": "⚙"},
        {"step": "M3", "title": "Tree models", "desc": "Decision Tree · Random Forest · AdaBoost",
         "endpoint": "models", "icon": "⬡"},
        {"step": "M4", "title": "Cluster & reduce", "desc": "K-Means, hierarchical, DBSCAN, PCA, t-SNE",
         "endpoint": "models", "icon": "◈"},
        {"step": "M5", "title": "Validate & report", "desc": "5-fold CV, ROC-AUC, every artefact",
         "endpoint": "reports", "icon": "▦"},
    ]

    # Honest segmentation facts — the real numbers, no invented cluster names.
    segmentation = [
        {"label": "Hierarchical clusters", "value": "3",
         "note": "Agglomerative and divisive methods agree on K = 3 student groups"},
        {"label": "Best silhouette", "value": "0.376",
         "note": "Divisive method — moderate separation, the clusters overlap"},
        {"label": "DBSCAN density groups", "value": "8",
         "note": "On CGPA · backlogs · internships, with 0% of records called noise"},
    ]

    return render_template(
        "home.html",
        total_records=TOTAL_RECORDS,
        total_columns=TOTAL_COLUMNS,
        placed_pct=PLACED_PCT,
        not_placed_pct=NOT_PLACED_PCT,
        placed_count=PLACED_COUNT,
        not_placed_count=NOT_PLACED_COUNT,
        missing_cells=MISSING_CELLS,
        best_accuracy=RF_ACCURACY,
        dt_accuracy=DT_ACCURACY,
        best_clusters=BEST_CLUSTERS,
        selected_features=SELECTED_FEATURES,
        total_models=TOTAL_MODELS,
        total_charts=total_charts,
        signals=_top_signals(),
        leaderboard=leaderboard,
        pipeline=pipeline,
        segmentation=segmentation,
        placed_medians=_placed_medians(),
        deploy=DEPLOY_METRICS,
        leakage_note=LEAKAGE_NOTE,
    )


@app.route("/about")
def about():
    return render_template("about.html")


# ---- Dataset preview (first N rows shown; full file via the button) ----

PREVIEW_ROWS = 500
_preview_cache = None


def dataset_preview(n=PREVIEW_ROWS):
    """Read only the first n rows of the raw CSV (cached) so the page stays light.

    Rows come back as plain JSON-safe Python values (int / float / str / None)
    so the page can hand them to JavaScript and render them in small pages.
    """
    global _preview_cache
    if _preview_cache is None:
        df = pd.read_csv(DATASET_PATH, nrows=n)
        with open(DATASET_PATH, "r", encoding="utf-8", errors="replace") as handle:
            total = max(sum(1 for _ in handle) - 1, 0)  # minus the header line
        columns = [str(c) for c in df.columns]
        frame = df.astype(object).where(pd.notna(df), None)
        _preview_cache = (columns, frame.values.tolist(), total)
    return _preview_cache


@app.route("/dataset")
def dataset():
    columns, rows, total_rows = dataset_preview()
    return render_template(
        "dataset.html",
        preview_columns=columns,
        preview_rows=rows,
        preview_shown=len(rows),
        preview_limit=PREVIEW_ROWS,
        total_rows=total_rows,
        total_records=TOTAL_RECORDS,
        total_columns=TOTAL_COLUMNS,
        placed_pct=PLACED_PCT,
        not_placed_pct=NOT_PLACED_PCT,
        placed_count=PLACED_COUNT,
        not_placed_count=NOT_PLACED_COUNT,
        missing_cells=MISSING_CELLS,
        total_models=TOTAL_MODELS,
    )


@app.route("/dataset/csv")
def dataset_csv():
    """Open the full raw CSV inline, or download it with ?download=1."""
    as_attachment = "download" in request.args
    return send_from_directory(
        DATASET_DIR,
        DATASET_FILE,
        mimetype="text/csv",
        as_attachment=as_attachment,
        download_name=DATASET_FILE,
    )


PREPROCESS_STAGES = [
    {
        "step": "01",
        "title": "Cleaning",
        "icon": "🧼",
        "tone": "tint-peach",
        "file": "clean_del_mean_model_M2.csv",
        "script": "clean_del_mean_model_missing_imputer_M2.py",
        "summary": "Duplicates checked and every missing value filled — mean / median for numbers and the most frequent value for categories.",
        "changes": [
            "duplicate rows checked — 0 found",
            "{:,} missing cells completed".format(MISSING_CELLS),
            "mean · median · model imputation compared",
            "{} incomplete columns finished".format(MISSING_COLUMNS),
        ],
    },
    {
        "step": "02",
        "title": "Encoding",
        "icon": "🔤",
        "tone": "tint-lilac",
        "file": "clean_label_encode_M2.csv",
        "script": "clean_label_encode_M2.py (+ one-hot, ordinal, target, embedding)",
        "summary": "The 8 text columns become numbers. Label encoding drives the main pipeline, while four other encoders are benchmarked side by side.",
        "changes": [
            "text → numbers",
            "label · one-hot · ordinal · target · embedding",
            "Gender, City, CollegeTier, Stream, Specialisation, Hostel, HistoryOfBacklogs, CGPA_Tier encoded",
        ],
    },
    {
        "step": "03",
        "title": "Scaling",
        "icon": "⚖️",
        "tone": "tint-mint",
        "file": "clean_minmax_stand_norma_M2.csv",
        "script": "clean_minmax_stand_norma_M2.py",
        "summary": "Every numeric feature is rescaled so that no single column dominates. The file keeps the originals beside their z-score, Min-Max and normalized versions so the methods can be compared.",
        "changes": [
            "z-score · min-max · normalisation",
            "mean ≈ 0, standard deviation ≈ 1",
            "keeps the unscaled values beside each scaled copy",
        ],
    },
]

FINAL_STAGE = {
    "step": "04",
    "title": "Final preprocessed file",
    "icon": "✅",
    "tone": "tint-lemon",
    "file": "final_preprocess_M2.csv",
    "script": "final_preprocess_M2.py",
    "summary": "One clean, fully numeric, scaled matrix — the single output that every model in the project reads.",
    "changes": [
        "{:,} rows × {} columns".format(TOTAL_RECORDS, TOTAL_COLUMNS),
        "0 missing values",
        "100% numeric & scaled",
        "saved as dataset/final_preprocess_M2.csv",
    ],
}

# The feature-selection study that follows the pipeline.
LASSO_STAGE = {
    "file": "lasso_feature_selection_M2.csv",
    "script": "Regularizatio_ridge_laso_elastic_M2.py",
}

ALLOWED_STAGE_FILES = (
    {stage["file"] for stage in PREPROCESS_STAGES}
    | {FINAL_STAGE["file"], LASSO_STAGE["file"]}
)

MAX_PREVIEW_COLS = 10
_stage_preview_cache = {}


def stage_preview(filename, n=6, max_cols=MAX_PREVIEW_COLS):
    """Preview the first n rows of a stage's CSV (cached, column-capped)."""
    key = (filename, n, max_cols)
    if key not in _stage_preview_cache:
        path = os.path.join(DATASET_DIR, filename)
        df = pd.read_csv(path, nrows=n)
        total_cols = df.shape[1]
        shown = [str(c) for c in df.columns[:max_cols]]
        data = df[shown].fillna("").values.tolist()
        _stage_preview_cache[key] = (shown, data, total_cols)
    return _stage_preview_cache[key]


@app.route("/preprocessing")
def preprocessing():
    def decorate(stage):
        columns, rows, total_cols = stage_preview(stage["file"])
        return {
            **stage,
            "columns": columns,
            "rows": rows,
            "total_cols": total_cols,
            "hidden_cols": max(total_cols - len(columns), 0),
        }

    _, _, total_rows = dataset_preview()
    return render_template(
        "preprocessing.html",
        stages=[decorate(stage) for stage in PREPROCESS_STAGES],
        final_stage=decorate(FINAL_STAGE),
        lasso_stage=decorate(LASSO_STAGE),
        total_rows=total_rows,
        total_columns=TOTAL_COLUMNS,
        missing_cells=MISSING_CELLS,
        selected_features=SELECTED_FEATURES,
    )


@app.route("/preprocessing/file/<path:filename>")
def preprocessing_file(filename):
    """Open or download one of the stage CSVs (?download=1 forces download)."""
    if filename not in ALLOWED_STAGE_FILES:
        abort(404)
    as_attachment = "download" in request.args
    return send_from_directory(
        DATASET_DIR,
        filename,
        mimetype="text/csv",
        as_attachment=as_attachment,
        download_name=filename,
    )


M1_FOLDERS = ["EDA_Analysis_outputs", "Boxplots_correlation"]


# ---- Visualisation: group the Module 1 charts into readable stories ----

VIZ_CATEGORIES = [
    ("target", "Placement relationships", "🎯",
     "How each feature behaves once students are split by placement outcome — the target we predict."),
    ("distributions", "Distributions", "📊",
     "Histograms that reveal the shape, skew and spread of every numeric feature."),
    ("boxplots", "Boxplots & spread", "📦",
     "Median, range and outliers per feature — overall and split by placement."),
    ("correlation", "Correlation & missing values", "🔥",
     "How the features move together, and exactly where the dataset has gaps."),
    ("counts", "Categorical counts", "🏷️",
     "How often each city, stream, college tier and specialisation appears."),
    ("patterns", "Outliers & patterns", "🧭",
     "Pairplots and scatter views used to spot structure and extreme points."),
    ("other", "Other diagnostics", "🖼️",
     "The remaining charts produced by the Module 1 analysis scripts."),
]


def viz_category(filename):
    """Map a chart filename to one of the visualisation categories."""
    name = filename.lower()
    if "vs_placementstatus" in name or "target" in name:
        return "target"
    if "correlation" in name or "missing" in name or "heatmap" in name:
        return "correlation"
    if "histogram" in name:
        return "distributions"
    if "boxplot" in name:
        return "boxplots"
    if "countplot" in name:
        return "counts"
    if any(word in name for word in ("pairplot", "scatter", "sctter", "outlier")):
        return "patterns"
    return "other"


# The charts the page opens with — the ones that set the scene.
VIZ_HIGHLIGHTS = [
    {
        "file": "EDA_Analysis_outputs/Correlation_Heatmap.png",
        "icon": "🔥",
        "tone": "tint-sky",
        "title": "Correlation heatmap",
        "fact": "Placement tracks Salary Package at r = 0.77 and the semester GPAs at 0.67–0.71 — "
                "strong links for student data. The catch is that one of those columns is only "
                "known after a student is placed.",
    },
    {
        "file": "Boxplots_correlation/boxplot_CGPA_vs_PlacementStatus.png",
        "icon": "📦",
        "tone": "tint-peach",
        "title": "Boxplot · CGPA vs placement",
        "fact": "The gap is real: placed students have a median CGPA of 8.02 against 5.74 for "
                "unplaced students, and the boxes barely overlap. Attendance splits the same way "
                "(81.9% vs 65.8%).",
    },
    {
        "file": "EDA_Analysis_outputs/Target_Distribution.png",
        "icon": "🎯",
        "tone": "tint-mint",
        "title": "Target balance",
        "fact": "65.7% of the 50,000 students are placed and 34.3% are not — a mild imbalance, "
                "which is why the project reports F1 and recall, not accuracy alone.",
    },
    {
        "file": "EDA_Analysis_outputs/Missing_Values_Heatmap.png",
        "icon": "🕳️",
        "tone": "tint-lilac",
        "title": "Missing-value map",
        "fact": "Exactly 5 columns have gaps — Workshops, AptitudeTestScore, SoftSkillsRating, "
                "CodingTestScore and MockInterviewScore (19,976 cells in total). "
                "Everything else is complete.",
    },
]


# Plain-language facts anyone can take away from the charts.
VIZ_FACTS = [
    ("🎓", "50,000 student records",
     "32 columns covering academics (CGPA, 8 semesters of GPA, attendance), skills (four test "
     "scores) and background (city, stream, college tier, hostel, backlogs)."),
    ("✅", "65.7% are placed",
     "32,856 students placed against 17,144 who were not — the target the whole project predicts."),
    ("🕳️", "Only five columns are incomplete",
     "19,976 missing cells in the workshop and test-score columns, all filled with the mean / "
     "median during cleaning. Every other column is complete."),
    ("📊", "Grades separate the two groups",
     "Median CGPA is 8.02 for placed students and 5.74 for unplaced ones — the clearest single "
     "signal in the charts."),
    ("🏫", "Ten cities, six streams",
     "Four specialisations and three college tiers are represented, so the data is not dominated "
     "by one campus or one course."),
    ("🚫", "One column gives it away",
     "Salary Package is above 0 for all 32,856 placed students and 0 for all but 68 of the "
     "unplaced ones. That single column explains the near-perfect model scores."),
    ("🌲", "Trees beat straight lines",
     "Because the signals interact (grades + tests + backlogs together), tree models beat a plain "
     "linear boundary — but the honest ceiling is set by how the data was created."),
]


def viz_highlights():
    """Only return highlight charts that actually exist on disk."""
    found = []
    for item in VIZ_HIGHLIGHTS:
        path = os.path.join(OUTPUTS_DIR, *item["file"].split("/"))
        if os.path.exists(path):
            found.append(item)
    return found


@app.route("/visualization")
def visualization():
    """Show every Module 1 exploratory chart, grouped into readable stories."""
    by_name = {m["name"]: m for m in _collect_modules()}
    modules = [by_name[folder] for folder in M1_FOLDERS if folder in by_name]

    charts = []
    for module in modules:
        for img in module["images"]:
            if img.get("group") != "charts":
                continue
            charts.append({**img, "cat": viz_category(img["name"])})

    meta = {key: (title, icon, blurb) for key, title, icon, blurb in VIZ_CATEGORIES}
    order = {key: i for i, (key, *_rest) in enumerate(VIZ_CATEGORIES)}

    sections = []
    for key in sorted(meta, key=lambda k: order[k]):
        items = sorted(
            (c for c in charts if c["cat"] == key),
            key=lambda c: c["name"].lower(),
        )
        if not items:
            continue
        title, icon, blurb = meta[key]
        sections.append({
            "key": key,
            "title": title,
            "icon": icon,
            "blurb": blurb,
            "charts": items,
            "count": len(items),
        })

    return render_template(
        "visualization.html",
        sections=sections,
        modules=modules,
        highlights=viz_highlights(),
        facts=VIZ_FACTS,
        total_images=len(charts),
        total_messages=sum(m["n_messages"] for m in modules),
    )


@app.route("/models")
def models():
    """List every model by name, with a jump/filter bar at the top."""
    by_name = {m["name"]: m for m in _collect_modules()}
    folders = [folder for spec in MODEL_GROUPS for folder in spec["folders"]]

    entries = []
    for folder in folders:
        module = by_name.get(folder)
        if not module:
            continue
        # Drop the "Module N · " prefix so only the model name shows.
        short = re.sub(r"^Module\s*\d+\s*[·\-]\s*", "", module["title"]).strip()
        entries.append({
            "key": folder,
            "short": short,
            "module": module,
            "n_images": module["n_images"],
            "n_messages": module["n_messages"],
        })

    return render_template(
        "models.html",
        models=entries,
        total_models=TOTAL_MODELS,
        total_columns=TOTAL_COLUMNS,
        comparison=CLASSIFICATION_COMPARISON,
        regression=REGRESSION_COMPARISON,
        unsupervised=UNSUPERVISED_COMPARISON,
        lasso_features=LASSO_FEATURES,
        overview=OVERVIEW,
        best_model=BEST_MODEL,
        baseline_accuracy=BASELINE_ACCURACY,
        baseline_f1=BASELINE_F1,
        deploy=DEPLOY_METRICS,
        leakage_note=LEAKAGE_NOTE,
        unlisted=UNLISTED_MODEL_FOLDERS,
    )


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    """Live placement prediction through the saved Decision Tree pipeline."""
    result = None
    error = None
    form = default_form_values()

    if request.method == "POST":
        for name in list(form):
            if name in request.form:
                form[name] = request.form.get(name, form[name]).strip()

        try:
            # StudentID, Salary Package and IsAnomaly are deliberately left out:
            # the pipeline fills them with the training median, so the form can
            # never be "cheated" by a value only known after a placement.
            row = {"StudentID": None, "Salary Package": None, "IsAnomaly": None}

            for group in FIELD_GROUPS:
                for field in group["fields"]:
                    raw = form.get(field["name"], "").strip()
                    if field["type"] == "select":
                        row[field["name"]] = raw or None
                    else:
                        row[field["name"]] = float(raw) if raw else None

            frame = pd.DataFrame([row])
            model = load_model()

            classes = list(model.named_steps["classifier"].classes_)
            placed_index = classes.index(1)

            prediction_value = int(model.predict(frame)[0])
            probability = float(model.predict_proba(frame)[0][placed_index])

            result = {
                "placed": prediction_value == 1,
                "probability": round(probability * 100, 1),
                "model_label": MODEL_LABEL,
                "signals": _gap_signals(row),
            }
        except Exception as exc:  # pragma: no cover - shown to the user as-is
            error = str(exc)

    return render_template(
        "prediction.html",
        field_groups=FIELD_GROUPS,
        form=form,
        result=result,
        error=error,
        deploy=DEPLOY_METRICS,
        leakage_note=LEAKAGE_NOTE,
    )


# ================================================================
# LIVE SCORING API (powers the Readiness Console on the dashboard)
#
# Reads the same trained pipeline as the Prediction page. Every field the
# caller leaves out is passed as missing, so the pipeline's imputers fill it
# with the training median — the same protection used everywhere else.
# ================================================================

GAP_FEATURES = [
    "CGPA", "AttendancePercent", "AptitudeTestScore", "CodingTestScore",
    "MockInterviewScore", "SoftSkillsRating", "Internships", "Projects",
]

# Columns that are only known AFTER a placement, or that simply identify a row.
LEAKY_FIELDS = {"Salary Package", "IsAnomaly"}

_columns_cache = None
_medians_cache = None
_signals_cache = None


def _dataset_columns():
    global _columns_cache
    if _columns_cache is None:
        _columns_cache = [str(c) for c in pd.read_csv(DATASET_PATH, nrows=0).columns]
    return _columns_cache


def _placed_medians():
    """Median value of each key feature among the students who WERE placed."""
    global _medians_cache
    if _medians_cache is None:
        try:
            frame = pd.read_csv(DATASET_PATH, usecols=["PlacementStatus"] + GAP_FEATURES)
            placed = frame[frame["PlacementStatus"] == 1]
            _medians_cache = {f: float(placed[f].median()) for f in GAP_FEATURES}
        except Exception:
            _medians_cache = {}
    return _medians_cache


def _top_signals(top=8):
    """Rank features by |correlation| with PlacementStatus, straight from EDA."""
    global _signals_cache
    if _signals_cache is None:
        try:
            corr = pd.read_csv(
                os.path.join(OUTPUTS_DIR, "EDA_Analysis_outputs", "Correlation_Matrix.csv"),
                index_col=0,
            )
            series = corr["PlacementStatus"].astype(float)
            series = series.drop(labels=["PlacementStatus"], errors="ignore")
            series = series.reindex(series.abs().sort_values(ascending=False).index)[:top]
            _signals_cache = [
                {
                    "name": str(name),
                    "r": round(float(value), 4),
                    "pct": round(abs(float(value)) * 100),
                    "flag": str(name).strip().lower() in ("salary package", "isanomaly"),
                }
                for name, value in series.items()
            ]
        except Exception:
            _signals_cache = []
    return _signals_cache


def _gap_signals(row):
    """Compare the caller's values with the placed-student median."""
    medians = _placed_medians()
    out = []
    for feature in GAP_FEATURES:
        if row.get(feature) is None or feature not in medians:
            continue
        value = float(row[feature])
        delta = value - medians[feature]
        out.append({
            "label": feature,
            "value": round(value, 2),
            "median": round(medians[feature], 2),
            "delta": round(delta, 2),
            "tone": "good" if delta >= 0 else "gap",
        })
    return out


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Return a placement probability plus gap analysis as JSON."""
    try:
        payload = request.get_json(silent=True) or {}
        row = {col: None for col in _dataset_columns()}

        for group in FIELD_GROUPS:
            for field in group["fields"]:
                raw = str(payload.get(field["name"], "")).strip()
                if not raw:
                    continue
                row[field["name"]] = raw if field["type"] == "select" else float(raw)

        model = load_model()
        classes = list(model.named_steps["classifier"].classes_)
        placed_index = classes.index(1)

        probability = float(model.predict_proba(pd.DataFrame([row]))[0][placed_index])
        prediction = int(model.predict(pd.DataFrame([row]))[0])

        return jsonify({
            "probability": round(probability * 100, 1),
            "placed": prediction == 1,
            "signals": _gap_signals(row),
            "note": (
                "Scored live by the leakage-free Random Forest — 91.5% accuracy and 93.7% F1 "
                "on 10,000 held-out students. It never sees Salary Package, so this estimate "
                "uses academics, skills and background only."
            ),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ================================================================
# RESULTS GALLERY (every chart + every generated message)
# ================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"}
TEXT_EXTS = {".txt", ".md", ".log", ".json"}

# Internal note/log/json files (like dataset_information.txt) are NOT shown on
# the website. The files stay on disk; only the display is switched off.
# Set this to True if they should appear again.
SHOW_TEXT_NOTES = False
TABLE_EXTS = {".csv", ".tsv"}

# Friendly titles / one-line descriptions / icon for each module folder.
MODULE_META = {
    "EDA_Analysis_outputs": (
        "Module 1 · Exploratory Data Analysis",
        "Histograms, counts, correlation, missing-value and outlier views of the raw student data.",
        "🔎",
    ),
    "Boxplots_correlation": (
        "Module 1 · Boxplots & Correlation",
        "Boxplot comparisons against the placement target plus the correlation heatmap.",
        "📦",
    ),
    "Linear_Regression_with_Metrics_M2": (
        "Module 2 · Linear Regression",
        "Predicting the placement number with every regression metric and diagnostic plot.",
        "📈",
    ),
    "Linear_Regression_CFNE_GD_Compare_M2": (
        "Module 2 · Normal Equation vs Gradient Descent",
        "Closed-form vs iterative solutions compared on loss and residuals.",
        "🧮",
    ),
    "Logistic_Regression_Binary_Classify_M2": (
        "Module 2 · Binary Logistic Regression",
        "Sigmoid, cross-entropy loss, confusion matrix and decision boundary.",
        "🔗",
    ),
    "Multinomial_Logistic_Regre_for_Multiclass_M2": (
        "Module 2 · Multinomial Logistic Regression",
        "Multi-class prediction on the CGPA tier with softmax probabilities.",
        "🧩",
    ),
    "Regularization_ridg_laso_elastic_outputs": (
        "Module 2 · Ridge / Lasso / Elastic Net",
        "L1 vs L2 shrinkage, sparsity and the feature-selection study.",
        "⚖️",
    ),
    "Decision_Tree_Classifier_M3_Outputs": (
        "Module 3 · Decision Tree",
        "Interpretable placement baseline: tree diagram, importance and metrics.",
        "🌳",
    ),
    "Random_Forest_Tree_M3_Outputs": (
        "Module 3 · Random Forest",
        "Best-scoring ensemble for placement, with a sample tree.",
        "🌲",
    ),
    "AdaBoost_Classfier_M3_Outputs": (
        "Module 3 · AdaBoost",
        "Boosting model trained on the 3-class CGPA tier target.",
        "🚀",
    ),
    "K_Means_K++Means_Elbow_Silhoute_M4_Outputs": (
        "Module 4 · K-Means & K-Means++",
        "Elbow and silhouette selection of the number of student clusters.",
        "🌀",
    ),
    "Hierarchical_Clust_Agglome_Divisive_M4_Outputs": (
        "Module 4 · Hierarchical Clustering",
        "Agglomerative vs divisive dendrograms and cluster assignments.",
        "🪜",
    ),
    "DBSCAN_Pre_Processed_dataset_M4_outputs": (
        "Module 4 · DBSCAN",
        "Density clustering on CGPA, backlogs and internships.",
        "🔷",
    ),
    "PCA_M4_Outputs": (
        "Module 4 · PCA",
        "Scree plot, cumulative variance, 2-D projection and PC1 loadings.",
        "📉",
    ),
    "t-SNE_M4_outputs": (
        "Module 4 · t-SNE",
        "Non-linear 2-D embedding of the numeric features.",
        "🌌",
    ),
    "CV_DT_All_Classifi_Regres_Metrics_M5_outputs": (
        "Module 5 · Cross-Validated Decision Tree",
        "5-fold stratified CV with ROC-AUC, PR-AUC and full metric reports.",
        "✅",
    ),
    "Deployment_Model_M6_Outputs": (
        "Module 6 · Deployment Model (leakage-free)",
        "The honest model the website serves: the leaky Salary Package column removed, 91.5% accuracy.",
        "🚀",
    ),
}

MODULE_ORDER = list(MODULE_META.keys())

# How the ML Models page orders the model folders.
MODEL_GROUPS = [
    {
        "title": "Module 3 — Tree models",
        "icon": "🌳",
        "blurb": "Decision Tree, Random Forest and AdaBoost placement classifiers.",
        "folders": [
            "Decision_Tree_Classifier_M3_Outputs",
            "Random_Forest_Tree_M3_Outputs",
            "AdaBoost_Classfier_M3_Outputs",
        ],
    },
    {
        "title": "Module 2 — Regression & regularisation",
        "icon": "📈",
        "blurb": "Linear regression, both solvers compared, logistic models and regularised coefficients.",
        "folders": [
            "Linear_Regression_with_Metrics_M2",
            "Linear_Regression_CFNE_GD_Compare_M2",
            "Logistic_Regression_Binary_Classify_M2",
            "Multinomial_Logistic_Regre_for_Multiclass_M2",
            "Regularization_ridg_laso_elastic_outputs",
        ],
    },
    {
        "title": "Module 4 — Clustering & dimensionality reduction",
        "icon": "🧩",
        "blurb": "K-Means, hierarchical and density clustering, with PCA and t-SNE.",
        "folders": [
            "K_Means_K++Means_Elbow_Silhoute_M4_Outputs",
            "Hierarchical_Clust_Agglome_Divisive_M4_Outputs",
            "DBSCAN_Pre_Processed_dataset_M4_outputs",
            "PCA_M4_Outputs",
            "t-SNE_M4_outputs",
        ],
    },
    {
        "title": "Module 5 — Cross-validated decision tree",
        "icon": "✅",
        "blurb": "5-fold stratified CV with every classification and regression metric.",
        "folders": [
            "CV_DT_All_Classifi_Regres_Metrics_M5_outputs",
        ],
    },
    {
        "title": "Module 6 — Deployment model",
        "icon": "🚀",
        "blurb": "The leakage-free Random Forest the website actually serves.",
        "folders": [
            "Deployment_Model_M6_Outputs",
        ],
    },
]

# Folders that exist but cannot be shown on the Models page, with the verified
# reason. Listed openly so the page can explain the gap instead of hiding it.
UNLISTED_MODEL_FOLDERS = [
    (
        "K-Means / K-Means++ (Module 4)",
        "the Elbow and Silhouette scripts ran, but wrote no chart or metric file "
        "into its output folder — it is empty.",
    ),
    (
        "Multinomial Logistic Regression (Module 2)",
        "produced only dataset_information.txt, and this site hides plain text notes, "
        "so there is nothing to display.",
    ),
]

# --------------------------------------------------------------------------
# Verified model comparison shown on the ML Models page.
#
# Every number below was read from the real output files, for example
# outputs/Random_Forest_Tree_M3_Outputs/metrics/random_forest_metrics.csv
#
# Target   : PlacementStatus (1 = placed, 0 = not placed)
# Test set : 10,000 rows, 80/20 split, random_state = 42
#            (identical split for every model, so the scores compare fairly)
# --------------------------------------------------------------------------

BEST_MODEL = {
    "name": "Random Forest",
    "headline": "Accuracy 99.82% · F1 99.82% — the highest of the placement models",
    "reasons": [
        "Highest accuracy and F1 of every model that predicts PlacementStatus.",
        "An ensemble of 100 trees, so it is far more stable than a single tree — "
        "one strange record cannot flip the answer.",
        "Reads the raw mixed dataset directly: median imputation and one-hot encoding "
        "happen inside its own pipeline.",
        "The Decision Tree matches it closely (99.77%) and is the model served on the "
        "Prediction page, because a single tree can be drawn and explained.",
    ],
}

# Ranked by F1, best first. accuracy=None means the run reported cross-validated
# metrics only (the M5 script). The final row is a reference, not a model.
CLASSIFICATION_COMPARISON = [
    {"rank": 1, "model": "Random Forest", "kind": "Ensemble", "best": True,
     "accuracy": 0.9982, "precision": 0.9982049172863863, "recall": 0.9982,
     "f1": 0.9981988625819868, "note": "BEST SCORE — 100 trees, depth-limited leaves"},
    {"rank": 2, "model": "Decision Tree", "kind": "Single tree",
     "accuracy": 0.9977, "precision": 0.997700910127964, "recall": 0.9977,
     "f1": 0.9976991142333107, "note": "max_depth 6 · served on the Prediction page"},
    {"rank": 3, "model": "CV Decision Tree (M5, 5-fold)", "kind": "Tree + cross-validation",
     "accuracy": None, "precision": 0.9977198869302318, "recall": 0.99772,
     "f1": 0.9977199363828438, "note": "ROC-AUC 0.9974 · PR-AUC 0.9976"},
    {"rank": "—", "model": "Always answer \"placed\" (reference)", "kind": "No model", "ref": True,
     "accuracy": BASELINE_ACCURACY, "precision": BASELINE_ACCURACY, "recall": 1.0,
     "f1": BASELINE_F1, "note": "The lazy baseline every real model must beat"},
]

# --------------------------------------------------------------------------
# Table 2 - Regression (target: PlacementStatus treated as a number)
# Read from outputs/Linear_Regression_with_Metrics_M2/linear_regression_metrics.csv
# --------------------------------------------------------------------------

REGRESSION_COMPARISON = [
    {"model": "Linear Regression", "mae": 0.5603, "mse": 0.4743, "rmse": 0.6887,
     "r2": 0.5253, "note": "Explains 52.5% of the variation — the most trustworthy number in the project"},
    {"model": "Normal Equation (closed form)", "mae": None, "mse": None, "rmse": None,
     "r2": None, "note": "Converged to the same solution — no metric file was saved"},
    {"model": "Gradient Descent (iterative)", "mae": None, "mse": None, "rmse": None,
     "r2": None, "note": "Loss curve saved; reaches the closed-form answer"},
]

# --------------------------------------------------------------------------
# Table 3 - Unsupervised learning and dimensionality reduction.
# No labels -> these models cannot predict placement.
# --------------------------------------------------------------------------

UNSUPERVISED_COMPARISON = [
    {"model": "Hierarchical — Agglomerative", "task": "Clustering",
     "result": "K = 3 · Silhouette 0.3612 · Calinski-Harabasz 281.38 · Davies-Bouldin 1.0022"},
    {"model": "Hierarchical — Divisive", "task": "Clustering",
     "result": "K = 3 · Silhouette 0.3759 · Calinski-Harabasz 332.59 · Davies-Bouldin 0.9700"},
    {"model": "DBSCAN", "task": "Density clustering",
     "result": "8 clusters · eps 0.80 · min_samples 10 · 0% noise · 50,000 rows used"},
    {"model": "PCA", "task": "Dimensionality reduction",
     "result": "19 components keep 95% of variance · PC1 = 46.40%"},
    {"model": "t-SNE", "task": "Dimensionality reduction",
     "result": "2-D embedding · perplexity 30 · KL divergence 2.5341"},
    {"model": "K-Means / K-Means++", "task": "Clustering",
     "result": "Elbow and silhouette scripts ran, but their output folders are empty — no artefacts saved"},
]

# --------------------------------------------------------------------------
# Table 4 - Lasso feature selection (12 of 31 features survived L1 shrinkage).
# Read from dataset/lasso_feature_selection_M2.csv
# --------------------------------------------------------------------------

LASSO_FEATURES = [
    {"feature": "Salary Package", "ridge": 0.5233, "lasso": 0.4118, "elastic": 0.4488, "selected": True},
    {"feature": "CGPA_Tier", "ridge": 0.2908, "lasso": 0.2367, "elastic": 0.2546, "selected": True},
    {"feature": "HistoryOfBacklogs", "ridge": -0.0918, "lasso": -0.0440, "elastic": -0.0683, "selected": True},
    {"feature": "SoftSkillsRating", "ridge": 0.0175, "lasso": 0.0289, "elastic": 0.0252, "selected": True},
    {"feature": "MockInterviewScore", "ridge": 0.0071, "lasso": 0.0232, "elastic": 0.0168, "selected": True},
    {"feature": "AptitudeTestScore", "ridge": 0.0335, "lasso": 0.0209, "elastic": 0.0267, "selected": True},
    {"feature": "ExtraCurricular", "ridge": -0.0407, "lasso": -0.0150, "elastic": -0.0260, "selected": True},
    {"feature": "Internships", "ridge": 0.0194, "lasso": 0.0140, "elastic": 0.0163, "selected": True},
    {"feature": "Projects", "ridge": 0.0189, "lasso": 0.0124, "elastic": 0.0148, "selected": True},
    {"feature": "CGPA", "ridge": -0.1106, "lasso": -0.0119, "elastic": -0.0504, "selected": True},
    {"feature": "Certifications", "ridge": 0.0015, "lasso": 0.0044, "elastic": 0.0035, "selected": True},
    {"feature": "AttendancePercent", "ridge": 0.0078, "lasso": 0.0031, "elastic": 0.0058, "selected": True},
]

# --------------------------------------------------------------------------
# Quick reference: every model and method in the project, with its own target.
# --------------------------------------------------------------------------

OVERVIEW = [
    {"model": "Linear Regression", "task": "Regression", "target": "PlacementStatus (as 0/1)",
     "result": "MAE 0.5603 · RMSE 0.6887 · R² 0.5253"},
    {"model": "Normal Equation vs Gradient Descent", "task": "Regression (2 methods)",
     "target": "PlacementStatus (as 0/1)", "result": "Both converge to the same solution · loss curve saved"},
    {"model": "Ridge / Lasso / Elastic Net", "task": "Regularisation (3 methods)",
     "target": "PlacementStatus", "result": "Lasso zeroes 19 of 31 features — see Table 4"},
    {"model": "Binary Logistic Regression", "task": "Classification", "target": "PlacementStatus",
     "result": "Saved predictions mix -1/0 actual with 0/1 predicted labels — no reliable score published"},
    {"model": "Multinomial Logistic Regression", "task": "Multi-class", "target": "CGPA_Tier (3 classes)",
     "result": "Runs on the 3-class target; only a description file was saved"},
    {"model": "Decision Tree", "task": "Classification", "target": "PlacementStatus",
     "result": "Accuracy 99.77% · F1 99.77% · max_depth 6"},
    {"model": "Random Forest", "task": "Ensemble", "target": "PlacementStatus",
     "result": "Accuracy 99.82% · F1 99.82% · best of all models"},
    {"model": "AdaBoost", "task": "Ensemble (boosting)", "target": "CGPA_Tier (3 classes)",
     "result": "Accuracy 99.96% · F1 99.96% — a different 3-class target, not placement"},
    {"model": "K-Means / K-Means++", "task": "Clustering", "target": "student profiles",
     "result": "Elbow and silhouette scripts ran; no artefact files were saved"},
    {"model": "Hierarchical — Agglomerative", "task": "Clustering", "target": "4 academic features",
     "result": "K = 3 · Silhouette 0.3612 · 500 sampled students"},
    {"model": "Hierarchical — Divisive", "task": "Clustering", "target": "4 academic features",
     "result": "K = 3 · Silhouette 0.3759 · beats the agglomerative version"},
    {"model": "DBSCAN", "task": "Density clustering", "target": "CGPA, backlogs, internships",
     "result": "8 clusters · eps 0.80 · min_samples 10 · 0% noise"},
    {"model": "PCA", "task": "Dimensionality reduction", "target": "32 numeric features",
     "result": "19 components for 95% variance · PC1 = 46.40%"},
    {"model": "t-SNE", "task": "Dimensionality reduction", "target": "32 numeric features",
     "result": "2-D embedding · perplexity 30 · KL 2.5341 · 1,000 iterations"},
    {"model": "CV Decision Tree (M5)", "task": "5-fold CV classification", "target": "PlacementStatus",
     "result": "F1 99.77% · ROC-AUC 0.9974 · PR-AUC 0.9976"},
    {"model": "Lasso feature selection", "task": "Feature selection", "target": "PlacementStatus",
     "result": "12 of 31 features kept"},
    {"model": "Deployment model (M6)", "task": "Classification (served live)",
     "target": "PlacementStatus", "result": "Leakage-free Random Forest · accuracy 91.52% · F1 93.68% · ROC-AUC 0.9766"},
]

# keep acronyms looking right when prettifying file names
_ACRONYMS = {
    "pca": "PCA", "dbscan": "DBSCAN", "tsne": "t-SNE", "sne": "t-SNE",
    "roc": "ROC", "auc": "ROC-AUC", "pr": "PR", "pc": "PC", "pc1": "PC1",
    "kmeans": "K-Means", "cv": "CV", "gd": "GD", "cfne": "CFNE",
    "l1": "L1", "l2": "L2", "id": "ID", "vs": "vs", "3d": "3-D", "2d": "2-D",
    "sgpa": "SGPA", "cgpa": "CGPA", "sem": "Sem",
}


def _prettify(filename):
    """Turn 'confusion_matrix.png' style names into a readable caption."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = re.sub(r"^\d+[_\- ]*", "", stem)
    text = re.sub(r"[_\-]+", " ", stem).strip()
    words = []
    for word in text.split():
        words.append(_ACRONYMS.get(word.lower(), word))
    caption = " ".join(words)
    return caption[:1].upper() + caption[1:] if caption else stem


def _fmt_cell(value):
    """Format a single table cell for display."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, float):
        if float(value).is_integer() and abs(value) < 1e12:
            return f"{int(value):,}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return str(value)


def _read_text(path, limit=7000):
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        data = handle.read(limit + 1)
    return data[:limit], len(data) > limit


def _count_data_rows(path, size_cap=30 * 1024 * 1024):
    """Count CSV data rows (header excluded) for small/medium files."""
    try:
        if os.path.getsize(path) > size_cap:
            return None
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            total = sum(1 for _ in handle)
        return max(total - 1, 0)
    except OSError:
        return None


def _read_table(path, max_rows=12):
    try:
        frame = pd.read_csv(path, nrows=max_rows)
    except Exception:
        return [], [], True
    headers = [str(col) for col in frame.columns]
    rows = [[_fmt_cell(v) for v in record] for record in frame.itertuples(index=False, name=None)]
    return headers, rows, len(rows) >= max_rows


def _scan_modules():
    """Walk every module output folder and gather images, messages and downloads."""
    modules = []

    if not os.path.isdir(OUTPUTS_DIR):
        return modules

    present = [
        name
        for name in os.listdir(OUTPUTS_DIR)
        if os.path.isdir(os.path.join(OUTPUTS_DIR, name)) and name != "ui_previews"
    ]
    ordered = [n for n in MODULE_ORDER if n in present]
    ordered += [n for n in sorted(present) if n not in MODULE_ORDER]

    for name in ordered:
        folder = os.path.join(OUTPUTS_DIR, name)
        images, messages, downloads = [], [], []

        for root, _dirs, names in os.walk(folder):
            for file_name in sorted(names):
                full = os.path.join(root, file_name)
                rel = os.path.relpath(full, OUTPUTS_DIR).replace("\\", "/")
                ext = os.path.splitext(file_name)[1].lower()
                sub = os.path.relpath(root, folder).replace("\\", "/")
                group = "charts" if sub == "." else sub.split("/")[0]

                if ext in IMAGE_EXTS:
                    images.append(
                        {
                            "rel": rel,
                            "name": file_name,
                            "caption": _prettify(file_name),
                            "group": group,
                        }
                    )
                elif ext in TEXT_EXTS:
                    if not SHOW_TEXT_NOTES:
                        continue
                    text, truncated = _read_text(full)
                    messages.append(
                        {
                            "rel": rel,
                            "name": file_name,
                            "group": group,
                            "kind": "text",
                            "text": text,
                            "truncated": truncated,
                        }
                    )
                elif ext in TABLE_EXTS:
                    headers, rows, truncated = _read_table(full)
                    messages.append(
                        {
                            "rel": rel,
                            "name": file_name,
                            "group": group,
                            "kind": "table",
                            "headers": headers,
                            "rows": rows,
                            "total": _count_data_rows(full),
                            "shown": len(rows),
                            "truncated": truncated,
                        }
                    )
                else:
                    downloads.append(
                        {
                            "rel": rel,
                            "name": file_name,
                            "group": group,
                            "ext": (ext.lstrip(".") or "file"),
                        }
                    )

        if not (images or messages or downloads):
            continue

        title, blurb, icon = MODULE_META.get(name, (name.replace("_", " "), "", "📦"))
        modules.append(
            {
                "name": name,
                "title": title,
                "blurb": blurb,
                "icon": icon,
                "images": images,
                "messages": messages,
                "downloads": downloads,
                "n_images": len(images),
                "n_messages": len(messages),
                "n_downloads": len(downloads),
            }
        )

    return modules


_modules_cache = None


def _collect_modules():
    """Cached module scan - the output folders do not change while running."""
    global _modules_cache
    if _modules_cache is None:
        _modules_cache = _scan_modules()
    return _modules_cache


@app.route("/reports")
def reports():
    """Show every generated chart and every generated message, per module."""
    modules = _collect_modules()

    total_images = sum(m["n_images"] for m in modules)
    total_messages = sum(m["n_messages"] for m in modules)
    total_downloads = sum(m["n_downloads"] for m in modules)

    return render_template(
        "reports.html",
        modules=modules,
        total_images=total_images,
        total_messages=total_messages,
        total_downloads=total_downloads,
        total_modules=len(modules),
    )


# ================================================================
# WARM-UP
# Build the cached previews now so the first page view is instant.
# ================================================================

def _warm_caches():
    try:
        dataset_preview()
        for stage in PREPROCESS_STAGES:
            stage_preview(stage["file"])
        stage_preview(FINAL_STAGE["file"])
        stage_preview(LASSO_STAGE["file"])
        _collect_modules()
    except Exception:
        pass  # never block startup on a cache miss


_warm_caches()


if __name__ == "__main__":
    # placement_prediction always runs on 5056 so it can never collide with
    # the CartIQ project, which uses 5055.
    app.run(debug=True, port=5056)
