# train_model.py
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    print("Train score:", clf.score(X_train, y_train))
    print("Test score:", clf.score(X_test, y_test))

    joblib.dump({
        "model": clf,
        "feature_names": data.feature_names,
        "target_names": list(data.target_names),
        "model_version": "1.0.0"
    }, "model.pkl")
    print("Saved model to model.pkl")

if __name__ == "__main__":
    main()
