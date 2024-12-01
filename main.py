from scripts.preprocess import load_and_preprocess_data
from scripts.sklearn_models import create_models, train_sklearn_model
from scripts.lstm_model import train_lstm_model
from scripts.evaluate import evaluate_model
import numpy as np

# File Path
file_path = './data/emotions.csv'

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

# Define label names
label_mapping = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}

# Train and evaluate Sklearn models
models = create_models()
for name, model in models.items():
    print(f"\n{name}:")
    trained_model = train_sklearn_model(model, X_train, y_train)
    evaluate_model(trained_model, X_test, y_test, target_names=list(label_mapping.values()))

# Train and evaluate LSTM model
X_train_lstm = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])
lstm_model, _ = train_lstm_model(X_train_lstm, y_train, input_shape=(1, X_train.shape[1]), num_classes=3)
evaluate_model(lstm_model, X_test_lstm, y_test, target_names=list(label_mapping.values()))
