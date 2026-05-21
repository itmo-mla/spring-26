from typing import Dict, List, Any, Tuple
import numpy as np
from itertools import product
from models.base import BaseEnsemble


class GridSearchOOB:
    
    def __init__(self, estimator: BaseEnsemble, param_grid: Dict[str, List[Any]], 
                 cv: int = 1, scoring: str = 'oob_score', n_jobs: int = -1):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GridSearchOOB':
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        best_score = -np.inf
        best_params = None
        best_estimator = None
        
        all_scores = []
        all_params = []

        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))

            for key, value in params.items():
                if value == "None" or (isinstance(value, str) and value.lower() == "none"):
                    params[key] = None

            estimator = type(self.estimator)(**params)
            estimator.oob_score = True

            estimator.fit(X, y)

            score = estimator._oob_score_ if estimator._oob_score_ is not None else -np.inf
            
            all_scores.append(score)
            all_params.append(params)
            
            if score > best_score:
                best_score = score
                best_params = params
                best_estimator = estimator
        
        self.best_score_ = best_score
        self.best_params_ = best_params
        self.best_estimator_ = best_estimator
        self.cv_results_ = {
            'params': all_params,
            'mean_test_score': all_scores
        }
        
        return self
