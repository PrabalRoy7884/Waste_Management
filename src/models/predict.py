import joblib
import pandas as pd

def load_model(model_path):
    return joblib.load(model_path)

def make_predictions(model, df):
    X = df.drop(columns=['Recycling Rate (%)'], errors='ignore')
    return model.predict(X)

def save_predictions(predictions, output_path):
    pd.DataFrame({'Predicted Recycling Rate (%)': predictions}).to_csv(output_path, index=False)
