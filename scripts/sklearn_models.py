from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_sklearn_model(model, X_train, y_train, param_grid=None):
    """
    Trains a scikit-learn model with optional hyperparameter tuning.

    Args:
        model: The scikit-learn model instance to train.
        X_train: Training features.
        y_train: Training labels.
        param_grid (dict, optional): Hyperparameters for grid search.

    Returns:
        Trained model.
    """
    if param_grid:
        # Use GridSearchCV for hyperparameter optimization
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    else:
        # Train the model without hyperparameter tuning
        model.fit(X_train, y_train)
        return model

def create_models():
    """
    Creates a dictionary of sklearn models.

    Returns:
        dict: Dictionary of model names and instances.
    """
    return {
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(solver='liblinear', C=1.0),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
