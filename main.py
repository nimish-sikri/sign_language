import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset (replace 'dataset.csv' with the actual path to your dataset)
data = pd.read_csv("dataset.csv")

# Check for null values in the output column
null_columns = data.columns[data.isnull().any()]
if "Output" in null_columns:
    print("Output column contains null values. Removing rows with null values...")
    data.dropna(subset=["Output"], inplace=True)

# Separate features (X) and target variable (y) after removing null values
X = data.drop(columns=["Output"])
y = data["Output"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize K-Nearest Neighbors classifier
knn = KNeighborsClassifier()

# Define hyperparameters to search
param_grid = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
}

# Perform grid search cross-validation to find the best hyperparameters
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Initialize the K-Nearest Neighbors classifier with the best hyperparameters
best_knn = KNeighborsClassifier(**best_params)

# Perform cross-validation
cv_scores = cross_val_score(best_knn, X, y, cv=5)  # 5-fold cross-validation

# Print cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))

# Train the classifier with the best hyperparameters on the entire training set
best_knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = best_knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks(np.arange(len(np.unique(y))), np.unique(y), rotation=45)
plt.yticks(np.arange(len(np.unique(y))), np.unique(y))
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
        )
plt.show()
