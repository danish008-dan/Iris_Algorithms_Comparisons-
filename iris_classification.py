# IRIS DATASET CLASSIFICATION

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
df = pd.read_csv("Iris.csv")   # Load Iris dataset
print("First five rows of dataset:")
print(df.head())
print("\nColumns:", df.columns.tolist())

# Step 2: Data cleaning / encoding
# Assuming Species column has text like 'Iris-setosa', etc.
df["Species"] = df["Species"].replace({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
})

# Verify mapping worked correctly
print("\nUnique class labels after encoding:", df["Species"].unique())

# Step 3: Feature and target split
X = df.drop("Species", axis=1)   # independent features
y = df["Species"]                      # target variable (0, 1, 2)

# Split dataset into training and testing parts (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Logistic Regression
log_model = LogisticRegression(max_iter=200)  # create model instance
log_model.fit(X_train, y_train)                # train the model
y_pred_log = log_model.predict(X_test)         # make predictions

print("\n=== Logistic Regression Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))
print("------------------------------------------------------")

#Step 5: K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)  # create model with k=5
knn_model.fit(X_train, y_train)                  # train the model
y_pred_knn = knn_model.predict(X_test)           # make predictions

print("=== K-Nearest Neighbors Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))
print("------------------------------------------------------")

# Step 6: Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)  # create model instance
dt_model.fit(X_train, y_train)                      # train the model
y_pred_dt = dt_model.predict(X_test)                # make predictions

print("=== Decision Tree Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))
print("------------------------------------------------------")

# Step 7: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 trees
rf_model.fit(X_train, y_train)                                        # train the model
y_pred_rf = rf_model.predict(X_test)                                  # make predictions

print("=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("------------------------------------------------------")

# Step 8: Compare model accuracies 
print("=== Model Accuracy Comparison ===")
print("Logistic Regression:", accuracy_score(y_test, y_pred_log))
print("KNN:", accuracy_score(y_test, y_pred_knn))
print("Decision Tree:", accuracy_score(y_test, y_pred_dt))
print("Random Forest:", accuracy_score(y_test, y_pred_rf))
