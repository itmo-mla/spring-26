import time
from pathlib import Path

import cornac
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD as SurpriseSVD
from surprise.model_selection import cross_validate

from data import download_movielens, build_matrix, train_test_split
from metrics import rmse_from_df, ndcg_at_k
from slim import SLIM
from svd_mf import SVD

PLOTS = Path(__file__).parent.parent / "plots"
PLOTS.mkdir(exist_ok=True)


print("Loading MovieLens 100K")
df = download_movielens()
print(f"Total ratings: {len(df)}")
print(f"Users: {df['user_id'].nunique()}")
print(f"Items: {df['item_id'].nunique()}")
print(f"Rating range: {df['rating'].min()} – {df['rating'].max()}")
print(f"Sparsity: {1 - len(df) / (df['user_id'].nunique() * df['item_id'].nunique()):.4%}")

train_df, test_df = train_test_split(df, test_ratio=0.2)
print(f"\nTrain / Test: {len(train_df)} / {len(test_df)}")

R_train, u2i, v2i = build_matrix(train_df)

print("SLIM (custom implementation, ElasticNet per column)")
t0 = time.perf_counter()
slim = SLIM(alpha=0.5, l1_ratio=0.5, max_iter=500)
slim.fit(R_train)
slim_time = time.perf_counter() - t0

slim_preds = slim.predict(R_train)
slim_rmse_train = rmse_from_df(train_df, slim_preds, u2i, v2i)
slim_rmse_test = rmse_from_df(test_df, slim_preds, u2i, v2i)
slim_ndcg = ndcg_at_k(slim_preds, R_train, k=10)

print(f"Train RMSE: {slim_rmse_train:.4f}")
print(f"Test  RMSE: {slim_rmse_test:.4f}")
print(f"NDCG@10: {slim_ndcg:.4f}")
print(f"Time: {slim_time:.1f}s")

print("EASE (cornac reference — item-item linear model, cf. SLIM)")
feedback = cornac.data.Dataset.from_uir(
    [(str(r.user_id), str(r.item_id), r.rating) for r in train_df.itertuples()],
    seed=42,
)
t0 = time.perf_counter()
ease_model = cornac.models.EASE(lamb=500, posB=True, verbose=False, seed=42)
ease_model.fit(feedback)
ref_slim_time = time.perf_counter() - t0

test_uids = [str(r.user_id) for r in test_df.itertuples()]
test_iids = [str(r.item_id) for r in test_df.itertuples()]
test_true = test_df["rating"].values.astype(float)

ref_ease_preds = []
for uid, iid in zip(test_uids, test_iids):
    try:
        p = ease_model.score(uid, iid)
    except Exception:
        p = feedback.global_mean
    ref_ease_preds.append(p)

ref_slim_rmse = float(np.sqrt(np.mean((test_true - np.array(ref_ease_preds)) ** 2)))
print(f"Test  RMSE: {ref_slim_rmse:.4f}")
print(f"Time: {ref_slim_time:.1f}s")

print("SVD Matrix Factorization (custom, SGD / Funk SVD)")
t0 = time.perf_counter()
svd = SVD(n_factors=50, n_epochs=20, lr=0.005, reg=0.02)
svd.fit(R_train)
svd_time = time.perf_counter() - t0

svd_preds = svd.predict_matrix()
svd_rmse_train = rmse_from_df(train_df, svd_preds, u2i, v2i)
svd_rmse_test = rmse_from_df(test_df, svd_preds, u2i, v2i)
svd_ndcg = ndcg_at_k(svd_preds, R_train, k=10)

print(f"Train RMSE: {svd_rmse_train:.4f}")
print(f"Test  RMSE: {svd_rmse_test:.4f}")
print(f"NDCG@10: {svd_ndcg:.4f}")
print(f"Time: {svd_time:.1f}s")

print("SVD (surprise reference, 5-fold CV)")
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)
t0 = time.perf_counter()
cv_results = cross_validate(
    SurpriseSVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02),
    surprise_data, measures=["RMSE"], cv=5, verbose=False,
)
ref_svd_time = time.perf_counter() - t0
ref_svd_rmse = float(cv_results["test_rmse"].mean())
print(f"CV RMSE: {ref_svd_rmse:.4f}")
print(f"Time: {ref_svd_time:.1f}s")


print("RESULTS SUMMARY")
summary = pd.DataFrame({
    "Model": ["SLIM (custom)", "EASE (cornac ref)", "SVD (custom)", "SVD (surprise)"],
    "Test RMSE": [slim_rmse_test, ref_slim_rmse, svd_rmse_test, ref_svd_rmse],
    "NDCG@10": [f"{slim_ndcg:.4f}", "—", f"{svd_ndcg:.4f}", "—"],
    "Time (s)": [f"{slim_time:.1f}", f"{ref_slim_time:.1f}",
                 f"{svd_time:.1f}", f"{ref_svd_time:.1f}"],
})
print(summary.to_string(index=False))


fig, ax = plt.subplots(figsize=(9, 5))
models = ["SLIM\n(custom)", "EASE\n(cornac ref)", "SVD\n(custom)", "SVD\n(surprise ref)"]
rmse_vals = [slim_rmse_test, ref_slim_rmse, svd_rmse_test, ref_svd_rmse]
colors = ["#4c72b0", "#c44e52", "#55a868", "#dd8452"]
bars = ax.bar(models, rmse_vals, color=colors, edgecolor="black", linewidth=0.7, width=0.55)
for bar, val in zip(bars, rmse_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Test RMSE", fontsize=12)
ax.set_title("RMSE Comparison: Custom vs Reference Implementations", fontsize=13)
ax.set_ylim(0, max(rmse_vals) * 1.15)
ax.grid(axis="y", linestyle="--", alpha=0.5)
from matplotlib.patches import Patch
legend = [Patch(color="#4c72b0", label="SLIM (custom)"),
          Patch(color="#c44e52", label="EASE (cornac ref)"),
          Patch(color="#55a868", label="SVD (custom)"),
          Patch(color="#dd8452", label="SVD (surprise ref)")]
ax.legend(handles=legend, fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(PLOTS / "rmse_comparison.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
epochs = range(1, len(svd.train_rmse_history) + 1)
ax.plot(epochs, svd.train_rmse_history, marker="o", markersize=4,
        color="#55a868", linewidth=2, label="Train RMSE")
ax.axhline(svd_rmse_test, linestyle="--", color="orange", linewidth=1.5,
           label=f"Test RMSE = {svd_rmse_test:.4f}")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("RMSE", fontsize=12)
ax.set_title("SVD (custom): Learning Curve", fontsize=13)
ax.legend(fontsize=10)
ax.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(PLOTS / "svd_learning_curve.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
ndcg_models = ["SLIM (custom)", "SVD (custom)"]
ndcg_vals = [slim_ndcg, svd_ndcg]
bars = ax.bar(ndcg_models, ndcg_vals, color=["#4c72b0", "#55a868"],
              edgecolor="black", linewidth=0.7, width=0.4)
for bar, val in zip(bars, ndcg_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("NDCG@10", fontsize=12)
ax.set_title("NDCG@10 Comparison (custom implementations)", fontsize=13)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(PLOTS / "ndcg_comparison.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
df["rating"].value_counts().sort_index().plot(kind="bar", ax=ax,
    color="#4c72b0", edgecolor="black", linewidth=0.7)
ax.set_xlabel("Rating", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("MovieLens 100K — Rating Distribution", fontsize=13)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(PLOTS / "rating_distribution.png", dpi=150)
plt.close()
