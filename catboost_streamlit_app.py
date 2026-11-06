import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="CatBoost Model Evaluation",
    page_icon="🌳",
    layout="wide"
)

st.title("🌳 CatBoost Model Evaluation Dashboard")
st.write(
    "Upload a CSV dataset, select the target column, train a CatBoost "
    "classifier and evaluate its performance."
)

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV file is empty.")
    st.stop()

st.subheader("1. Dataset preview")
st.write(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")
st.dataframe(df.head(10), use_container_width=True)

columns = df.columns.tolist()

default_target_index = columns.index("label") if "label" in columns else len(columns) - 1
target_col = st.selectbox(
    "Select the target column (the value you want the model to predict)",
    columns,
    index=default_target_index
)

# Exclude likely identifiers and direct target proxies from the features.
default_excluded = []
for col in columns:
    if col.lower() in {"timestamp", "date", "datetime", "id"}:
        default_excluded.append(col)

# In this water-leakage dataset, condition duplicates the label.
if target_col.lower() == "label" and "condition" in columns:
    default_excluded.append("condition")
if target_col.lower() == "condition" and "label" in columns:
    default_excluded.append("label")

default_excluded = list(dict.fromkeys(
    col for col in default_excluded if col != target_col
))

st.subheader("2. Select model inputs")
st.caption(
    "Choose columns that the model may use to make predictions. "
    "Do not include columns that reveal the answer, such as a duplicate "
    "of the target label."
)

feature_options = [col for col in columns if col != target_col]
selected_features = st.multiselect(
    "Feature columns",
    feature_options,
    default=[col for col in feature_options if col not in default_excluded]
)

if not selected_features:
    st.error("Select at least one feature column.")
    st.stop()

# Keep only selected features and target, then remove rows with missing target.
work = df[selected_features + [target_col]].copy()
work = work.dropna(subset=[target_col])

if work[target_col].nunique() != 2:
    st.error(
        f"This dashboard currently evaluates binary classification. "
        f"The selected target '{target_col}' has {work[target_col].nunique()} "
        "unique classes. Select a target column with exactly two classes."
    )
    st.stop()

# Convert categorical columns to strings and use an explicit missing category.
X = work[selected_features].copy()
y = work[target_col].copy()

categorical_features = []
for col in X.columns:
    if (
        pd.api.types.is_object_dtype(X[col])
        or isinstance(X[col].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(X[col])
    ):
        X[col] = X[col].fillna("__MISSING__").astype(str)
        categorical_features.append(col)

# CatBoost handles missing numeric values natively.
for col in X.columns:
    if col not in categorical_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

class_values = list(pd.unique(y))
st.write("**Target classes detected:**", ", ".join(map(str, class_values)))

positive_class = st.selectbox(
    "Select the positive class for Precision, Recall and F1 Score",
    options=class_values,
    index=1
)

negative_class = next(value for value in class_values if value != positive_class)

st.subheader("3. Model settings")
col1, col2, col3 = st.columns(3)

with col1:
    test_size_percent = st.slider(
        "Test dataset size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    )

with col2:
    iterations = st.number_input(
        "Number of iterations",
        min_value=50,
        max_value=2000,
        value=500,
        step=50
    )

with col3:
    depth = st.slider(
        "Tree depth",
        min_value=3,
        max_value=10,
        value=6
    )

learning_rate = st.select_slider(
    "Learning rate",
    options=[0.01, 0.03, 0.05, 0.1, 0.2],
    value=0.05
)

random_state = st.number_input(
    "Random state (for repeatable results)",
    min_value=0,
    max_value=99999,
    value=42
)

test_size = test_size_percent / 100

# Ensure stratified splitting is possible.
class_counts = y.value_counts()
n_test = int(np.ceil(len(y) * test_size))
n_train = len(y) - n_test

if class_counts.min() < 2:
    st.error("Each target class needs at least two rows for a stratified train/test split.")
    st.stop()

if n_test < 2 or n_train < 2:
    st.error("Not enough rows for the selected test size.")
    st.stop()

if st.button("🚀 Train CatBoost Model and Evaluate", type="primary", use_container_width=True):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=int(random_state),
            stratify=y
        )

        model = CatBoostClassifier(
            iterations=int(iterations),
            learning_rate=float(learning_rate),
            depth=int(depth),
            loss_function="Logloss",
            verbose=False,
            random_seed=int(random_state),
            allow_writing_files=False
        )

        with st.spinner("Training CatBoost model..."):
            model.fit(
                X_train,
                y_train,
                cat_features=categorical_features
            )
            predictions = model.predict(X_test).ravel()

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, pos_label=positive_class, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, pos_label=positive_class, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, pos_label=positive_class, zero_division=0
        )

        st.session_state["catboost_results"] = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "predictions": predictions,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "positive_class": positive_class,
            "negative_class": negative_class,
            "target_col": target_col,
            "feature_names": selected_features,
            "test_count": len(X_test),
            "train_count": len(X_train)
        }

    except Exception as e:
        st.error(f"Model training failed: {e}")

results = st.session_state.get("catboost_results")

if results is not None:
    # Do not show old results if the user has uploaded a different dataset
    # or changed model inputs/settings. The user can retrain to refresh them.
    st.divider()
    st.subheader("4. Model evaluation results")
    st.write(
        f"Training samples: **{results['train_count']}**  |  "
        f"Testing samples: **{results['test_count']}**"
    )
    st.caption(
        f"Positive class used for Precision, Recall and F1 Score: "
        f"{results['positive_class']}"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{results['accuracy'] * 100:.2f}%")
    m2.metric("Precision", f"{results['precision'] * 100:.2f}%")
    m3.metric("Recall", f"{results['recall'] * 100:.2f}%")
    m4.metric("F1 Score", f"{results['f1'] * 100:.2f}%")

    st.subheader("5. Confusion matrix")
    cm = confusion_matrix(
        results["y_test"],
        results["predictions"],
        labels=[results["negative_class"], results["positive_class"]]
    )
    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual {results['negative_class']}",
            f"Actual {results['positive_class']}"
        ],
        columns=[
            f"Predicted {results['negative_class']}",
            f"Predicted {results['positive_class']}"
        ]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("6. Classification report")
    report = classification_report(
        results["y_test"],
        results["predictions"],
        zero_division=0,
        output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    st.subheader("7. Feature importance")
    importance_df = pd.DataFrame({
        "Feature": results["feature_names"],
        "Importance": results["model"].get_feature_importance()
    }).sort_values("Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))

    st.subheader("8. Test-set predictions")
    predictions_df = results["X_test"].copy()
    predictions_df["Actual"] = results["y_test"].values
    predictions_df["Predicted"] = results["predictions"]
    predictions_df["Correct"] = (
        predictions_df["Actual"].astype(str)
        == predictions_df["Predicted"].astype(str)
    )
    st.dataframe(predictions_df, use_container_width=True)

    csv_data = predictions_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download test predictions (CSV)",
        data=csv_data,
        file_name="catboost_test_predictions.csv",
        mime="text/csv"
    )

    # Generate a concise, data-driven conclusion for the report.
    st.subheader("9. Evaluation conclusion")
    conclusion = (
        f"The CatBoost classification model was evaluated using a test "
        f"dataset containing {results['test_count']} samples. The model "
        f"achieved an accuracy of {results['accuracy'] * 100:.2f}%, "
        f"precision of {results['precision'] * 100:.2f}%, recall of "
        f"{results['recall'] * 100:.2f}%, and an F1 Score of "
        f"{results['f1'] * 100:.2f}%. Precision measures how many predicted "
        f"{results['positive_class']} cases were correct, while recall "
        f"measures how many actual {results['positive_class']} cases were "
        f"identified. The F1 Score summarises the balance between precision "
        f"and recall. These results provide an indication of the model's "
        f"performance on the held-out test data."
    )
    st.write(conclusion)
    st.download_button(
        "⬇️ Download evaluation conclusion (TXT)",
        data=conclusion,
        file_name="catboost_evaluation_conclusion.txt",
        mime="text/plain"
    )

st.caption(
    "Research note: A single train/test split can produce variable results, "
    "especially with a small dataset. For a stronger evaluation, consider "
    "stratified cross-validation and test the final model on independent data."
)
