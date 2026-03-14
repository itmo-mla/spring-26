import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

pd.set_option('display.max_columns', 100)
df = pd.read_csv("online_course.csv")
df = df.drop(columns=['UserID'])

for col in df.columns:
    if col != 'CourseCompletion':
        mask = np.random.random(len(df)) < 0.1
        df.loc[mask, col] = np.nan

cdf = df
cdf['CourseCategory'] = cdf['CourseCategory'].fillna('MISSING').astype(str)

y = cdf['CourseCompletion']
X = cdf.drop(columns=['CourseCompletion'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from catboost import CatBoostClassifier, Pool
categorical_features = ['CourseCategory']
train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=categorical_features
)
test_pool = Pool(
    data=X_test,
    cat_features=categorical_features
)
model = CatBoostClassifier(
    iterations=1,
    depth=8,
    learning_rate=1,
    verbose=False
)
model.fit(train_pool)
pred = model.predict(test_pool)

print("Библиотечная реализация")
print("accuracy:", accuracy_score(pred, y_test))
print("f1_score:", f1_score(pred, y_test))

import PreparingDataset
y = df['CourseCompletion']
X = df.drop(columns=['CourseCompletion'])
X['DeviceType'] = X['DeviceType'].astype(str)
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp,
    test_size=0.2,
    random_state=42,
    stratify=y_tmp
)

X_train['target'] = y_train
X_val['target'] = y_val
X_test['target'] = y_test
target = 'target'
train, porogs = PreparingDataset.nums_to_cat(X_train, target)
val = PreparingDataset.nums_to_cat_test(X_val, target, porogs)
test = PreparingDataset.nums_to_cat_test(X_val, target, porogs)

real = test[target]
test = test.drop(columns=[target])

df_combined = pd.concat([train, val], ignore_index=True)

import Gain
from Gain import Node, ListNode

head = Gain.builder(train, target, Node())
preds = []
for i in range(len(test)):
    ans = Gain.get_prediction(head, test.iloc[i])
    preds.append(ans[0])
preds = [int(x) for x in preds]
print("Моя реализация без pruning не используя вообще валидационные данные")
print("accuracy:", accuracy_score(preds, real))
print("f1_score:", f1_score(preds, real))


head = Gain.builder(df_combined, target, Node())
preds = []
for i in range(len(test)):
    ans = Gain.get_prediction(head, test.iloc[i])
    preds.append(ans[0])
preds = [int(x) for x in preds]
print("Моя реализация без pruning, используя валидационные данные pruninga для обучения!")
print("accuracy:", accuracy_score(preds, real))
print("f1_score:", f1_score(preds, real))

head = Gain.builder(train, target, Node())
Gain.predict_and_fill(val, head)
Gain.pruning_tree(head, target)
preds = []
for i in range(len(test)):
    ans = Gain.get_prediction(head, test.iloc[i])
    preds.append(ans[0])
preds = [int(x) for x in preds]
print("Моя реализация с pruning")
print("accuracy:", accuracy_score(preds, real))
print("f1_score:", f1_score(preds, real))
