from flask import Flask, request, render_template, redirect, url_for, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import pickle

app = Flask(__name__)

# File upload directory
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Step 1: Upload input files
        merged_file = request.files["merged_file"]
        test_file = request.files["test_file"]

        # Save uploaded files
        merged_path = os.path.join(UPLOAD_FOLDER, merged_file.filename)
        test_path = os.path.join(UPLOAD_FOLDER, test_file.filename)
        merged_file.save(merged_path)
        test_file.save(test_path)

        # Step 2: Process input data
        df = pd.read_excel(merged_path)
        test_data = pd.read_excel(test_path)

        # Train Random Forest to get optimal threshold
        gene_expression_columns = df.iloc[:, 1:4539].columns
        X = df[gene_expression_columns]
        X.columns = X.columns.astype(str)  # Ensure feature names are strings
        y = df['Responder_Label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)

        # Predict probabilities
        y_probs_rf = rf_model.predict_proba(X_test)[:, 1]
        fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_probs_rf)
        optimal_idx_rf = np.argmax(tpr_rf - fpr_rf)
        optimal_threshold_rf = thresholds_rf[optimal_idx_rf]

        # Reclassify responders
        df['Predicted_Responder'] = (df['IC50 (uM)'] <= optimal_threshold_rf).astype(int)

        # Find common genes
        train_genes = df.columns[1:4539]
        test_genes = test_data.columns[:414]
        common_genes = train_genes.intersection(test_genes)

        # KNN predictions for each common gene
        if len(common_genes) > 0:
            knn_model = KNeighborsClassifier(n_neighbors=5)
            knn_model.fit(df[common_genes], df['Predicted_Responder'])

            predictions = {}
            for gene in common_genes:
                X_gene_test = test_data[[gene]]
                predictions[gene] = knn_model.predict(X_gene_test)

            # Store predictions
            for gene, pred in predictions.items():
                test_data[f"{gene}_Predicted_Responder_KNN"] = pred

            results_path = os.path.join(RESULTS_FOLDER, "test_predictions.csv")
            test_data.to_csv(results_path, index=False)

            return redirect(url_for("results", filename="test_predictions.csv"))
        else:
            return "No common genes found between datasets."

    return render_template("index.html")

@app.route("/results/<filename>")
def results(filename):
    return send_file(os.path.join(RESULTS_FOLDER, filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
