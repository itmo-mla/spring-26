import os
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

SLIM_REPO = Path(__file__).resolve().parent / "SLIM"
PYTHON_PACKAGE = SLIM_REPO / "python-package"


def _native_lib_path() -> Path | None:
    systype = "Darwin" if platform.system().startswith("Darwin") else "Linux"
    arch = platform.machine()
    ext = "dylib" if systype == "Darwin" else "so"
    path = SLIM_REPO / "build" / f"{systype}-{arch}" / "src" / "libslim" / f"libslim.{ext}"
    return path if path.is_file() else None


def is_reference_slim_available() -> bool:
    return _native_lib_path() is not None


def _ensure_importable():
    lib = _native_lib_path()
    if lib is None:
        raise RuntimeError(
            "Эталонный SLIM не собран. Запустите из каталога source:\n"
            "  bash build_reference_slim.sh"
        )
    os.environ.setdefault("SLIM_LIB_PATH", str(lib))
    pkg = str(PYTHON_PACKAGE)
    if pkg not in sys.path:
        sys.path.insert(0, pkg)


def _import_slim():
    _ensure_importable()
    from SLIM import SLIM, SLIMatrix  # noqa: WPS433

    return SLIM, SLIMatrix


class ReferenceSLIM:
    def __init__(
        self,
        l1_reg: float = 1.0,
        l2_reg: float = 1.0,
        nthreads: int = 2,
        max_iter: int = 100,
        opt_tol: float = 1e-7,
    ):
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.nthreads = nthreads
        self.max_iter = max_iter
        self.opt_tol = opt_tol
        self._model = None
        self.W: csr_matrix | None = None

    def fit(self, R: csr_matrix):
        SLIM, SLIMatrix = _import_slim()

        if not isinstance(R, csr_matrix):
            R = csr_matrix(R)

        trainmat = SLIMatrix(R.astype(np.float32))
        params = {
            "algo": "cd",
            "nthreads": self.nthreads,
            "l1r": self.l1_reg,
            "l2r": self.l2_reg,
            "niters": self.max_iter,
            "optTol": self.opt_tol,
        }

        self._model = SLIM()
        self._model.train(params, trainmat)
        self.W = self._model.to_csr()
        return self

    def predict(self, R: csr_matrix) -> csr_matrix:
        if self.W is None:
            raise ValueError("Модель не обучена: вызовите fit().")
        if not isinstance(R, csr_matrix):
            R = csr_matrix(R)
        return (R @ self.W).tocsr()
