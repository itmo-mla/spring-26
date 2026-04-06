from tree.decision_tree import DecisionTree
from tree.pruning import prune_tree
from dataset import load_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


X_train, X_test, y_train, y_test = load_dataset()
tree = DecisionTree(max_depth=10)
tree.fit(X_train, y_train)
pred = tree.predict(X_test)
print("Custom tree")
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1:", f1_score(y_test, pred))


# pruning
prune_tree(tree, X_test, y_test)
pred_pruned = tree.predict(X_test)
print("\nAfter pruning")
print("Accuracy:", accuracy_score(y_test, pred_pruned))
print("Precision:", precision_score(y_test, pred_pruned))
print("Recall:", recall_score(y_test, pred_pruned))
print("F1:", f1_score(y_test, pred_pruned))


# sklearn
sk_tree = DecisionTreeClassifier(criterion="gini", max_depth=10)
sk_tree.fit(X_train, y_train)
pred_sk = sk_tree.predict(X_test)
print("\nSklearn tree")
print("Accuracy:", accuracy_score(y_test, pred_sk))
print("Precision:", precision_score(y_test, pred_sk))
print("Recall:", recall_score(y_test, pred_sk))
print("F1:", f1_score(y_test, pred_sk))