Running data pipeline...
Detected 0 categorical columns: []
Detected 13 numerical columns: ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline ']
Data pipeline finished in 1.78 sec
Data loaded: X_train=(125, 13), X_test=(53, 13)

============================================================
TRAINING CUSTOM GMM
============================================================
Custom GB: GaussianMixtureModel[fitted=False; n_components=3; max_iter=100; tol=0.0001]
Model trained in 0.029 sec
Evaluation:
 Log-likelihood: -982.29

============================================================
TRAINING SKLEARN GAUSSIAN MIXTURE
============================================================
Sklearn GB: GaussianMixture(init_params='random', n_components=3, random_state=42,
                tol=0.0001)
Model trained in 0.010 sec
Evaluation:
 Log-likelihood: -1004.44

============================================================
PIPELINE COMPLETE
============================================================