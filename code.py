import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =========================
# 1. 데이터 로드
# =========================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# =========================
# 2. 파생 변수 생성
# =========================
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 100],
    labels=["0-12", "13-24", "25-48", "49+"]
)

df["charge_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)
df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

# customerID 제거
df = df.drop(columns=["customerID"])

X = df.drop(columns=["Churn"])
y = df["Churn"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 3. 평가 함수
# =========================
def evaluate_model(name, pipeline):
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
    else:
        auc = None

    return {
        "Experiment": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-score": f1_score(y_test, pred),
        "ROC-AUC": auc
    }

results = []

# =========================
# Exp-1: Mean + OneHot + Standard
# =========================
preprocess_exp1 = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42)
}

for model_name, model in models.items():
    pipe = Pipeline([
        ("preprocess", preprocess_exp1),
        ("model", model)
    ])
    results.append(evaluate_model(f"Exp-1_{model_name}", pipe))

# =========================
# Exp-2: Median + Label Encoding + MinMax + Feature Selection
# =========================
preprocess_exp2 = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ]), cat_cols)
])

for model_name, model in models.items():
    pipe = Pipeline([
        ("preprocess", preprocess_exp2),
        ("select", SelectKBest(score_func=f_classif, k=15)),
        ("model", model)
    ])
    results.append(evaluate_model(f"Exp-2_{model_name}", pipe))

# =========================
# Exp-3: Most Frequent + OneHot + Robust + Feature Selection
# =========================
preprocess_exp3 = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", RobustScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

for model_name, model in models.items():
    pipe = Pipeline([
        ("preprocess", preprocess_exp3),
        ("select", SelectKBest(score_func=f_classif, k=20)),
        ("model", model)
    ])
    results.append(evaluate_model(f"Exp-3_{model_name}", pipe))

# =========================
# 결과 출력
# =========================
result_df = pd.DataFrame(results)
print(result_df)
