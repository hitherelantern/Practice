import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.01, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.initial_prediction = None

    def fit(self, X, y):
        # Initialize the model with the mean of the target values (regression setup)
        self.initial_prediction = np.mean(y)
        predictions = np.full_like(y, self.initial_prediction, dtype=np.float64)
        
        for _ in range(self.n_estimators):
            # Compute residuals (errors)
            residuals = y - predictions
            
            # Fit a weak learner (decision tree) to the residuals
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X, residuals)
            
            # Make predictions using the current model
            residual_pred = model.predict(X)
            
            # Update the predictions using the learning rate
            predictions += self.learning_rate * residual_pred
            
            # Store the model for later use during prediction
            self.models.append(model)
    
    def predict(self, X):
        # Start with the initial prediction
        predictions = np.full(X.shape[0], self.initial_prediction, dtype=np.float64)
        
        # Add contributions from all weak learners
        for model in self.models:
            residual_pred = model.predict(X)
            predictions += self.learning_rate * residual_pred
        
        return predictions



if __name__ == "__main__":

    # Generate a binary classification dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the Gradient Boosting model
    gb = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)
    gb.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = gb.predict(X_test)

    # Convert continuous predictions to binary predictions (0 or 1)
    binary_predictions = (predictions > 0.5).astype(int)

    # Evaluate the model
    accuracy = accuracy_score(y_test, binary_predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
