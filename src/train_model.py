import numpy as np
import pickle
from data_preprocessing import preprocess_data

# Helper function to calculate regression metrics
def calculate_metrics(y_true, y_pred):
    # mean_squared_error = 1/n * sum(y_true - y_pred)^2
    mse = np.mean((y_true - y_pred) ** 2)

    # root_mean_squared_error = (mean_squared_error)^0.5
    rmse = np.sqrt(mse)

    # total_sum_of_squares = sum(y_true - mean(y_true))^2
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)

    # residual_sum_of_squares = sum(y_true - y_pred)^2
    ss_residual = np.sum((y_true - y_pred) ** 2)

    # r2_score = 1 - (residual_sum_of_squares / total_sum_of_squares)
    r2 = 1 - (ss_residual / ss_total)

    return mse, rmse, r2

class LinearRegression:
    # Linear Regression class
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Here we impelement Gradient Descent
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # y = wx + b
        return np.dot(X, self.weights) + self.bias

class L2Regression(LinearRegression):
    # L2 Regularization class. Inherits from LinearRegression.
    def __init__(self, learning_rate=0.001, n_iterations=1000, l=1.0):
        super().__init__(learning_rate, n_iterations)
        self.l = l  # Regularization strength (lambda)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            # y = w.x + b
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients with L2 regularization term
            # dw = (1 / n_samples) * (X.T . (y_predicted - y)) + 2 * lambda * w)
            dw = (1 / n_samples) * (np.dot(X.T, (y_predicted - y)) + 2 * self.l * self.weights)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

class L1Regression(LinearRegression):
    # L1 Regularization class. Inherits from LinearRegression.
    def __init__(self, learning_rate=0.001, n_iterations=1000, l=1.0):
        super().__init__(learning_rate, n_iterations)
        self.l = l  # Regularization strength (lambda)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Add L1 regularization term to gradients
            dw += self.l * np.sign(self.weights)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


# Model Training script with k fold cross validation
# Trains multiple regression models and saves them to models dir.
def cross_validate(model, X, y, k=5):
    indices = np.arange(X.shape[0])
    # Shuffling the indices to ensure random splits
    np.random.shuffle(indices)

    # Split the shuffled indices into k buckets (folds)
    fold_indices = np.array_split(indices, k)

    fold_scores = []

    # Iterate through each fold
    for i in range(k):
        # The ith bucket is the validation set
        val_indices = fold_indices[i]

        # The other k-1 buckets are the training set
        train_indices = np.concatenate([fold_indices[j] for j in range(k) if j != i])

        # Slice the data based on indices
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Train the model on the training data
        model.fit(X_train, y_train)

        # Predict on the validation data
        y_pred = model.predict(X_val)

        # Calculate metrics for the fold
        _, rmse, r2 = calculate_metrics(y_val, y_pred)
        fold_scores.append(rmse)
        print(f"  Fold {i+1}/5 RMSE: {rmse:.4f} R^2: {r2:.4f}")

    # 4. Calculate the average score across all folds 
    average_rmse = np.mean(fold_scores)
    return average_rmse


def train(data_path, models_dir):
    """
    Uses cross-validation to find the best model, then retrains it on all
    data and saves all experimented models.
    """
    print("Loading and preprocessing data...")
    X, y = preprocess_data(data_path, feature_key='avg_purchase_value')
    print("Data loaded and preprocessed.")

    # Convert to NumPy arrays for the models
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    # Model config for experimentation
    model_configs = {
        "LinearRegression": LinearRegression(learning_rate=0.009, n_iterations=500),
        "L2_l0.01": L2Regression(learning_rate=0.1, n_iterations=500, l=0.02),
        "L2_l0.1": L2Regression(learning_rate=0.01, n_iterations=1000, l=0.3),  
        "L1_l0.01": L1Regression(learning_rate=0.01, n_iterations=1000, l=1),
    }

    best_model_name = None
    best_model_instance = None
    best_model_score = float('inf') # Initialize with infinity

    print("Starting Cross Validation")
    for name, model in model_configs.items():
        print(f"\nEvaluating {name}...")
        avg_rmse = cross_validate(model, X_np, y_np, k=5)
        print(f"Average RMSE for {name}: {avg_rmse:.4f}")

        if avg_rmse < best_model_score:
            best_model_score = avg_rmse
            best_model_name = name
            best_model_instance = model

    print(f"\nCross Validation Complete")
    print(f"Best model found: {best_model_name} with an average RMSE of {best_model_score:.4f}")

    # Re train the Best Model on the full dataset to avoid overfitting
    print(f"\nRe training the best model ({best_model_name}) on all data...")
    best_model_instance.fit(X_np, y_np)

    # Save All Models
    print("\nSaving models")
    model_files = ["regression_model1.pkl", "regression_model2.pkl", "regression_model3.pkl"]
    other_models = {k: v for k, v in model_configs.items() if k != best_model_name}
    
    # Save the other models
    for (name, model), filename in zip(other_models.items(), model_files):
        model.fit(X_np, y_np) # Retrain on all data
        with open(f"{models_dir}/{filename}", 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {name} to {filename}")

    # Save the best model as 'final'
    with open(f"{models_dir}/regression_model_final.pkl", 'wb') as f:
        pickle.dump(best_model_instance, f)
    print(f"Saved best model ({best_model_name}) to regression_model_final.pkl")


if __name__ == '__main__':
    train('data/train_data.csv', 'models')