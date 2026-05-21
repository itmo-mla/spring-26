from src.model import GradientBoosting
import pandas as pd
import numpy as np
df = pd.read_csv("../lab1/data/Titanic-Dataset.xls")
target = "Survived"
df["Sex"] = np.where(df["Sex"] == 'female', 1, 0)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, stratify=y)

from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoosting(max_depth=10, n_estimators=50)
model.fit(x_train, y_train)
print(accuracy_score(y_pred=model.predict(x_test), y_true=y_test))
accuracy_score(y_pred=GradientBoostingClassifier(max_depth=10, n_estimators=50).fit(x_train, y_train).predict(x_test), y_true=y_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42, )

clf = GradientBoosting(max_depth=10, n_estimators=50)

cv_res = cross_validate(clf, x_train, y_train, cv=kf, scoring=['accuracy'], n_jobs=-1)


cv_res[f"test_accuracy"].mean()


kf = KFold(n_splits=5, shuffle=True, random_state=42, )

clf = GradientBoostingClassifier(max_depth=10, n_estimators=50)

cv_res = cross_validate(clf, x_train, y_train, cv=kf, scoring=['accuracy'], n_jobs=-1)

cv_res[f"test_accuracy"].mean()



