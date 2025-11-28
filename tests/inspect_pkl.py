# python
import joblib, pprint
model_path = r"C:\Users\Bernardo Carlo\Documents\python_ojd\model\plate_model_manual_only.pkl"
obj = joblib.load(model_path)
print("type:", type(obj))
if isinstance(obj, dict):
    print("keys:")
    pprint.pprint(list(obj.keys()))
    # common metadata keys
    if "meta" in obj:
        print("meta:")
        pprint.pprint(obj["meta"])
    if "training_history" in obj:
        print("training_history:")
        pprint.pprint(obj["training_history"])
else:
    # maybe a sklearn Pipeline or estimator
    print("repr:", repr(obj)[:1000])