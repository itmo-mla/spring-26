import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("../lab1/data/Titanic-Dataset.xls")
target = "Survived"
df["Sex"] = np.where(df["Sex"] == df["Sex"][0], 1, 0)
x, y = df.drop(columns=['target']).to_numpy(), df[target].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 0.33],
}

best_score = -1
best_params = None

for n_est, depath, feat in product(*param_grid.values()):
    model = RFClassifier(
        n_estimators=n_est, max_depth=depth,
        max_features=feat, min_samples_split=split,
        random_state=42
    )
    model.fit(X_train, y_train)
    if model.oob_score_ > best_score:
        best_score = model.oob_score_
        best_params = {'n_estimators': n_est, 'max_depth': depth, 
                       'max_features': feat}

print(f"{best_params}")
print(f"{best_score}")

custom_rf = RFClassifier(**best_params, random_state=42)
start = time.time()
custom_rf.fit(X_train, y_train)
custom_time = time.time() - start

custom_pred = custom_rf.predict(X_test)
custom_acc = accuracy_score(y_test, custom_pred)
custom_importances = custom_rf._compute_feature_importances(X_train, y_train)

sklearn_rf = SklearnRF(**best_params, random_state=42, oob_score=True, n_jobs=1)
start = time.time()
sklearn_rf.fit(X_train, y_train)
sklearn_time = time.time() - start

sklearn_pred = sklearn_rf.predict(X_test)
sklearn_acc = accuracy_score(y_test, sklearn_pred)

print("\n📊 Сравнение:")
print(f"{'Метрика':<20} {'Custom':>10} {'sklearn':>10}")
print(f"{'Accuracy (test)':<20} {custom_acc:>10.4f} {sklearn_acc:>10.4f}")
print(f"{'OOB Score':<20} {custom_rf.oob_score_:>10.4f} {sklearn_rf.oob_score_:>10.4f}")
print(f"{'Время обучения (с)':<20} {custom_time:>10.3f} {sklearn_time:>10.3f}")

print("\nВажность признаков (Custom RF, OOB^j):")
for i, imp in zip(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'], custom_importances):
    print(f"  Признак {i}: {imp:.4f}")