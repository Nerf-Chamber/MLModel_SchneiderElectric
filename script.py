import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# === NEW ===
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt


def testML():
    # Load training dataset
    df = pd.read_csv("dataset.csv")

    # Drop ID and extract target
    X = df.drop(["target_variable", "id"], axis=1)
    y = df["target_variable"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    log_reg = LogisticRegression(max_iter=300)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    # Stacking
    stack = StackingClassifier(
        estimators=[("xgb", xgb)],
        final_estimator=log_reg
    )

    # Train
    stack.fit(X_train_scaled, y_train)

    # Predict test set
    y_pred = stack.predict(X_test_scaled)

    print("=== Classification Report (Stacking) ===")
    print(classification_report(y_test, y_pred))

    # ---------------------------
    # Feature Importance (from XGBoost)
    # ---------------------------
    xgb_model = stack.named_estimators_["xgb"]

    print("\n=== Feature Importance (XGBoost) ===")
    importances = xgb_model.feature_importances_
    for feature, score in sorted(
        zip(X.columns, importances), key=lambda x: x[1], reverse=True
    ):
        print(f"{feature}: {score:.4f}")

    # ============================================================
    #                     SHAP EXPLAINABILITY
    # ============================================================

    print("\n=== SHAP Global Explainability ===")

    # Use SHAP TreeExplainer on the XGBoost model
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_train)

    # ---- GLOBAL SHAP PLOT (SUMMARY PLOT) ----
    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values, X_train, feature_names=X.columns, show=False)
    plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: shap_summary_plot.png")

    # ---- LOCAL SHAP example for a single case (first test row) ----
    print("\n=== SHAP Local Explanation for first test instance ===")
    instance_index = 0
    instance = X_test.iloc[instance_index:instance_index+1]

    shap.force_plot(
        explainer.expected_value,
        explainer.shap_values(instance),
        instance,
        feature_names=X.columns,
        matplotlib=True,
        show=False
    )
    plt.savefig("shap_local_case.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: shap_local_case.png")

    instance_index = 0
    instance = X_test.iloc[instance_index:instance_index + 1]
    shap_values_local = explainer.shap_values(instance)

    explicacions = interpretar_shap_local(shap_values_local, instance, X.columns)

    print("\n=== Explicació automàtica SHAP ===")
    for frase in explicacions:
        print("•", frase)

    # ============================================================
    #                     LIME EXPLAINABILITY
    # ============================================================

    print("\n=== LIME Local Explanation ===")

    lime = LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=X.columns,
        class_names=["0", "1"],
        mode="classification"
    )

    explanation = lime.explain_instance(
        X_test_scaled[instance_index],
        stack.predict_proba,
        num_features=10
    )

    explanation.save_to_file("lime_local_explanation.html")
    print("Saved: lime_local_explanation.html")

    # Retrieve LIME row example
    original_index = X_test.index[0]
    original_row = df.loc[original_index]
    print(f"LIME Row case = {original_row}")

    # ---------------------------
    # Predict from user-provided CSV
    # ---------------------------
    print("\n=== CSV Prediction Mode ===")
    csv_name = input("Enter the name of the CSV file to predict (Must be in the root of the project): ")

    input_df = pd.read_csv(csv_name)

    if "id" not in input_df.columns:
        raise ValueError("The input CSV must contain an 'id' column.")

    X_new = input_df.drop(["id"], axis=1)
    X_new_scaled = scaler.transform(X_new)

    predictions = stack.predict(X_new_scaled)
    probabilities = stack.predict_proba(X_new_scaled)[:, 1]

    input_df["predicted_target"] = predictions
    input_df["probability_of_1"] = probabilities

    print("\n=== Predictions for CSV ===")
    print(input_df[["id", "predicted_target", "probability_of_1"]])

    output_name = csv_name.replace(".csv", "_predictions.csv")
    input_df.to_csv(output_name, index=False)
    print(f"\nPredictions saved to: {output_name}")

def interpretar_shap_local(shap_values, instance, feature_names, threshold=0.05):
    explicacions = []

    for i, valor in enumerate(shap_values[0]):
        nom = feature_names[i]
        direccio = "incrementa" if valor > 0 else "redueix"
        intensitat = abs(valor)

        if intensitat >= threshold:
            frase = f"La característica '{nom}' {direccio} la probabilitat de guanyar amb un impacte de {valor:.3f}."
            explicacions.append(frase)

    return explicacions

if __name__ == "__main__":
    testML()
