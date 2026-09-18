import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap

# Load sample data (from CSV)
def load_sample_data():
    df = pd.read_csv("data/sample_defects/labels.csv")
    X = df.drop(['defect_class'], axis=1)
    y = df['defect_class']
    return X, y

# Train a simple causal classifier (e.g., on machine temp, shift, humidity)
def train_causal_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    print("Causal Model Accuracy:", model.score(X_test, y_test))
    return model

# SHAP explainer
def explain_causal(model, X_test, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return shap_values

# Example: get top 3 causes for a defect
def get_top_causes(model, X_row, feature_names, top_n=3):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    top_features = [(feature_names[i], feature_importance[i]) for i in sorted_idx[:top_n]]
    return top_features
