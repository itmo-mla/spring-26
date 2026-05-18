import importlib
import pkgutil
import inspect
from typing import Type, Dict
from data.dataset import Dataset

DATASETS: Dict[str, Type[Dataset]] = {}

package = __name__
for _, module_name, _ in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f"{package}.{module_name}")

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Dataset) and obj is not Dataset:
            key = ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_').replace('_dataset', '')
            DATASETS[key] = obj