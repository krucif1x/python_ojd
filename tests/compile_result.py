import sys
import pathlib
# Ensure project root is on sys.path so `import src...` works when running the script directly
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# adjust paths if needed
MODEL_PATH = r"C:\Users\Bernardo Carlo\Documents\python_ojd\model\plate_model_manual_only.pkl"
LABELS_CSV = r"C:\Users\Bernardo Carlo\Documents\python_ojd\data\manual_labels.csv"

# load saved dict
m = joblib.load(MODEL_PATH)
print("Loaded model keys:", list(m.keys()))

vec = m["vectorizer"]
prov_enc = m["province_encoder"]
reg_enc = m["region_encoder"]
veh_enc = m["vehicle_encoder"]
prov_clf = m["province_classifier"]
reg_clf = m["region_classifier"]
veh_clf = m["vehicle_classifier"]

# load CSV - adapt column names if different
df = pd.read_csv(LABELS_CSV)
texts = df["plate"].astype(str).tolist()
y_prov = df["province"].astype(str).tolist()
y_reg = df["region"].astype(str).tolist()
y_veh = df["vehicle_type"].astype(str).tolist()

# create features (tfidf + handcrafted). If your pipeline used extra features,
# import extractor to compute them the same way:
from src.nlp.nlp_metadata_extractor import ManualLabelsPlateExtractor
ext = ManualLabelsPlateExtractor(manual_labels_path=LABELS_CSV)
X_tfidf = vec.transform(texts).toarray()
X_hand = np.vstack([ext.extract_features(t) for t in texts])
X = np.hstack([X_tfidf, X_hand])

# vehicle
yv_pred = veh_clf.predict(X)
yv_pred_lbl = veh_enc.inverse_transform(yv_pred)
print("\n=== Vehicle classification ===")
print("Accuracy:", accuracy_score(y_veh, yv_pred_lbl))
print(classification_report(y_veh, yv_pred_lbl, zero_division=0))

# province
yp_pred = prov_clf.predict(X)
yp_pred_lbl = prov_enc.inverse_transform(yp_pred)
print("\n=== Province classification ===")
print("Accuracy:", accuracy_score(y_prov, yp_pred_lbl))
print(classification_report(y_prov, yp_pred_lbl, zero_division=0))

# region
yr_pred = reg_clf.predict(X)
yr_pred_lbl = reg_enc.inverse_transform(yr_pred)
print("\n=== Region classification ===")
print("Accuracy:", accuracy_score(y_reg, yr_pred_lbl))
print(classification_report(y_reg, yr_pred_lbl, zero_division=0))

# Optional: confusion matrix example for vehicle
print("\nVehicle confusion matrix (rows=true, cols=pred):")
print(pd.DataFrame(confusion_matrix(y_veh, yv_pred_lbl, labels=sorted(df["vehicle_type"].unique())),
                   index=sorted(df["vehicle_type"].unique()),
                   columns=sorted(df["vehicle_type"].unique())))