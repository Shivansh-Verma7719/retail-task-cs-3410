import pandas as pd
import pickle
import argparse
from data_preprocessing import preprocess_data
from train_model import L2Regression, LinearRegression, L1Regression, calculate_metrics

def main(model_path, data_path, metrics_output_path, predictions_output_path):
    """
    Loads a model, makes predictions, and saves metrics and predictions.
    """
    # Load the trained model
    with open(model_path, 'rb') as f:
        # print(f)
        model = pickle.load(f)

    # Preprocess the data
    X, y = preprocess_data(data_path, feature_key='avg_purchase_value')

    # Convert to numpy for prediction
    X_test = X.to_numpy()
    y_true = y.to_numpy()

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse, rmse, r2 = calculate_metrics(y_true, y_pred)

    # Save metrics to a text file
    with open(metrics_output_path, 'w') as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        f.write(f"R-squared (R^2) Score: {r2:.2f}\n")
    print(f"Metrics saved to {metrics_output_path}")

    # Save predictions to a CSV file
    pd.DataFrame(y_pred).to_csv(predictions_output_path, header=False, index=False)
    print(f"Predictions saved to {predictions_output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--metrics_output_path', type=str, required=True)
    parser.add_argument('--predictions_output_path', type=str, required=True)

    args = parser.parse_args()

    main(args.model_path, args.data_path, args.metrics_output_path, args.predictions_output_path)