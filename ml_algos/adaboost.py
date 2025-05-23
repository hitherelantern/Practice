import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score



class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []


    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y, sample_weight=w)
            predictions = model.predict(X)
            # Error of the current weak learner
            err = np.sum(w * (predictions != y)) / np.sum(w)
            # confidence score for the current weak learner/model (logit)
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
            self.alphas.append(alpha)
            self.models.append(model)
            # Rescale weights for the next iteration using the confidence score(higher logit = lower weight)
            w = w * np.exp(-alpha * y * predictions)
            # Normalize weights
            w = w / np.sum(w)
            
            


    def predict(self, X):
        strong_preds = np.zeros(X.shape[0])
        for model, alpha in zip(self.models, self.alphas):
            # Each weak model's prediction is weighted by its confidence score
            # and added to the strong predictions
            strong_preds += alpha * model.predict(X)
        # The final prediction is the sign of the strong predictions
        return np.sign(strong_preds).astype(int)
        


if __name__ == "__main__":


    # # Generate a multi-class classification dataset
    # X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative= 5, random_state=42)

    # # Split dataset into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # # Initialize AdaBoost with multi-class support
    # adaboost = AdaBoost(n_estimators=50)
    # adaboost.fit(X_train, y_train)

    # # Make predictions on the test set
    # predictions = adaboost.predict(X_test)

    # # Evaluate the performance of the classifier
    # accuracy = accuracy_score(y_test, predictions)
    # precision = precision_score(y_test, predictions, average='weighted')  # Weighted average for multi-class
    # recall = recall_score(y_test, predictions, average='weighted')  # Weighted average for multi-class
    # f1 = f1_score(y_test, predictions, average='weighted')  # Weighted average for multi-class

    # # ROC-AUC is undefined for multi-class without probability scores, so we will skip it for now
    # try:
    #     roc_auc = roc_auc_score(y_test, predictions, multi_class='ovr')  # One-vs-Rest for multi-class
    # except ValueError:
    #     roc_auc = 'Undefined (requires probability scores)'

    # # Print the results
    # print(f"Accuracy: {accuracy * 100}%")
    # print(f"Precision (weighted): {precision}")
    # print(f"Recall (weighted): {recall}")
    # print(f"F1 Score (weighted): {f1}")
    # print(f"ROC-AUC: {roc_auc}")



    # # Example values for w, predictions, and y
    # w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ,0.1 ,0.1 ,0.1])
    # predictions = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    # y = np.array([-1, -1, -1, 1, 1, -1, 1, -1, 1, -1])

    # # Calculate and print the value
    # error = np.sum(w * (predictions != y)) / np.sum(w)
    # print(f"Error: {error}")

    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    adaboost = AdaBoost(n_estimators=50)
    adaboost.fit(X_train, y_train)

    predictions = adaboost.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    try:
        roc_auc = roc_auc_score(y_test, predictions)
    except ValueError:
        roc_auc = 'Undefined (requires probability scores)'

    # Print results
    print(f"Accuracy: {accuracy * 100}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC-AUC: {roc_auc}")

