# Advanced: Machine Learning Classifier Example
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load sample dataset
data = load_iris()
X, y = data.data, data.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Model trained successfully!")
print(f"Accuracy: {accuracy:.2f}")

# Predict for a single sample
sample = X_test[0].reshape(1, -1)
predicted_class = model.predict(sample)[0]
print(f"Example prediction → Class: {data.target_names[predicted_class]}")
