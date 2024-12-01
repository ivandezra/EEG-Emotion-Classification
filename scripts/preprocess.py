import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the dataset for training and testing.

    Args:
        file_path (str): Path to the CSV dataset.

    Returns:
        X_train, X_test, y_train, y_test: Preprocessed train-test split data.
    """
    # Load data into a DataFrame
    data = pd.read_csv(file_path)
    
    # Map labels (categorical) to numerical values for machine learning
    label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
    data['label'] = data['label'].map(label_mapping)
    
    # Normalize features to ensure consistent scales
    scaler = StandardScaler()
    data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])
    
    # Split dataset into features (X) and target (y)
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Further split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
