import pandas as pd

# Standardizes the features using Z-score normalization.
def scaler(X):
    print(f"Starting Z-score normalization for {len(X.columns)} columns...")
    X_scaled = X.copy()
    
    # Iterating over each col
    for i, column in enumerate(X_scaled.columns, 1):
        mean = X_scaled[column].mean()
        std = X_scaled[column].std()
        
        print(f"Processing column {i}/{len(X_scaled.columns)}: '{column}' (mean={mean:.4f}, std={std:.4f})")
        
        # Avoiding division by 0
        if std != 0:
            X_scaled[column] = (X_scaled[column] - mean) / std
        else:
            # If std is 0 all values in the column are the same. Scaling results in 0.
            X_scaled[column] = 0
            print(f"Column '{column}' has zero standard deviation setting to 0")
    
    print("Z-score normalization completed!\n")
    return X_scaled

# Performs one-hot encoding for categorical variables
def one_hot_encode(X):
    print(f"Starting one-hot encoding...")
    
    # Identify categorical columns (object type)
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    print(f"Found {len(categorical_cols)} categorical columns: {categorical_cols}")
    print(f"Found {len(numerical_cols)} numerical columns")
    
    if len(categorical_cols) == 0:
        print("No categorical columns found, returning original dataframe")
        return X
    
    # Create a copy of the dataframe
    X_encoded = X.copy()
    
    # Define threshold for maximum unique values to one-hot encode
    max_unique_threshold = 10  # Only encode columns with <= 50 unique values
    
    # Separate categorical columns into encodable and non-encodable
    encodable_cols = []
    high_cardinality_cols = []
    
    for col in categorical_cols:
        unique_count = X_encoded[col].nunique()
        if unique_count <= max_unique_threshold:
            encodable_cols.append(col)
        else:
            high_cardinality_cols.append(col)
    
    print(f"Columns suitable for one-hot encoding ({len(encodable_cols)}): {encodable_cols}")
    print(f"High cardinality columns to drop ({len(high_cardinality_cols)}): {high_cardinality_cols}")
    
    # Drop high cardinality columns (like dates, IDs, etc.)
    if high_cardinality_cols:
        print("Dropping high cardinality categorical columns...")
        for col in high_cardinality_cols:
            unique_count = X_encoded[col].nunique()
            print(f"  Dropping '{col}' with {unique_count} unique values")
            X_encoded = X_encoded.drop(col, axis=1)
    
    # Apply one-hot encoding to suitable categorical columns
    if encodable_cols:
        print("Applying one-hot encoding to suitable categorical columns...")
        for i, col in enumerate(encodable_cols, 1):
            print(f"Encoding column {i}/{len(encodable_cols)}: '{col}'")
            unique_values = X_encoded[col].nunique()
            print(f"  Unique values in '{col}': {unique_values}")
            
            # Get dummies for this column
            dummies = pd.get_dummies(X_encoded[col], prefix=col, drop_first=True)
            print(f"  Created {len(dummies.columns)} dummy variables for '{col}'")
            
            # Drop original column and add dummy columns
            X_encoded = X_encoded.drop(col, axis=1)
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
    else:
        print("No categorical columns suitable for one-hot encoding")
    
    print(f"One-hot encoding completed!")
    print(f"Original shape: {X.shape}, Encoded shape: {X_encoded.shape}")
    
    return X_encoded

# Loads and preprocesses the life expectancy data.
def preprocess_data(data_path, feature_key):    
    # Load data
    print("Loading CSV data...")
    df = pd.read_csv(data_path)

    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Separate features (X) and target (y)
    X = df.drop(feature_key, axis=1)
    y = df[feature_key]

    # Apply one-hot encoding to categorical variables
    print("Starting one-hot encoding for categorical variables...")
    X_encoded = one_hot_encode(X)

    # Scaling numerical features
    print("Starting feature scaling...")
    X_scaled = scaler(X_encoded)
    
    print("Data preprocessing completed successfully!")
    return X_scaled, y

# For testing only
if __name__ == '__main__':
    # This is relative from base dir (I ran it from main dir)
    X, y = preprocess_data('data/train_data.csv', feature_key='avg_purchase_value')
    print("Data preprocessed successfully")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape, "\n")
    print("First 5 rows of features (X):")
    print(X.head())