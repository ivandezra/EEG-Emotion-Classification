from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, X_test, y_test, target_names):
    """
    Evaluates a trained model on test data.

    Args:
        model: Trained model instance.
        X_test: Test features.
        y_test: Test labels.
        target_names (list): Names of the target classes.
    """
    if hasattr(model, 'predict_proba'):
        # For sklearn models with predict_proba
        y_pred = np.argmax(model.predict_proba(X_test), axis=1)
    elif hasattr(model, 'predict'):
        # For TensorFlow models
        y_pred = np.argmax(model.predict(X_test), axis=1)
    else:
        y_pred = model.predict(X_test)

    # Print classification report
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("Classification Report:\n", report)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
