from dataset_bringer import AdultDataset
from tree import Tree
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

x_train, x_test, y_train, y_test = AdultDataset.get_dataset_split()
x_train, x_test, y_train, y_test = x_train[:300], x_test[:300], y_train[:300], y_test[:300]
print(f"Dataset ready test size {x_test.shape}")
tree = Tree()
tree_pruning = Tree(max_depth=30)
sklearn_tree = DecisionTreeClassifier(ccp_alpha=0)
sklearn_tree_pruning = DecisionTreeClassifier(ccp_alpha=0.005, max_depth=30)

tree.train(x_train, y_train)
print("Tree train finish")
tree_pruning.train(x_train, y_train, ccp_alpha=0.005)
print("Tree pruning train finish")
sklearn_tree_pruning.fit(x_train, y_train)
print("Sklearn tree pruning train finish")
sklearn_tree.fit(x_train, y_train)
print("Sklearn tree train finish")

y_pred = tree.predict(x_test)
y_pred_pruning = tree_pruning.predict(x_test)
y_pred_sklearn = sklearn_tree.predict(x_test)
y_pred_sklearn_pruning = sklearn_tree_pruning.predict(x_test)


report = classification_report(y_test, y_pred)
report_pruning = classification_report(y_test, y_pred_pruning)
report_sklearn = classification_report(y_test, y_pred_sklearn)
report_sklearn_pruning = classification_report(y_test, y_pred_sklearn_pruning)

print("="*20)
print("OWN")
print("="*20)
print(report)

print("="*20)
print("SKLEARN")
print("="*20)
print(report_sklearn)

print("="*20)
print("OWN PRUNING")
print("="*20)
print(report_pruning)

print("="*20)
print("SKLEARN PRUNING")
print("="*20)
print(report_sklearn)
