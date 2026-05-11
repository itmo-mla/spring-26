import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from source.boosting import GradientBoostingBinaryClassifier
from source.data import load_heart_dataset, prepare_dataset
from source.metrics import ModelBenchmark, evaluate_classification, pretty_metrics, stratified_cv_score


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def build_custom_model() -> GradientBoostingBinaryClassifier:
    return GradientBoostingBinaryClassifier(
        n_estimators=120,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )


def build_sklearn_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        loss="log_loss",
        n_estimators=120,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )


def save_bar_plot(output_path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 520
    margin = 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bar_area_h = height - 2 * margin
    bar_area_w = width - 2 * margin
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1e-9)
    bar_width = bar_area_w / max(1, len(values) * 2)
    baseline_y = height - margin

    draw.text((margin, 18), title, fill="black", font=title_font)
    draw.text((18, margin + bar_area_h / 2), ylabel, fill="black", font=font)
    draw.line((margin, margin, margin, baseline_y), fill="black", width=2)
    draw.line((margin, baseline_y, width - margin, baseline_y), fill="black", width=2)

    for idx, (label, value) in enumerate(zip(labels, values)):
        bar_left = margin + idx * bar_width * 2 + 10
        bar_height = bar_area_h * (value / max_value)
        bar_top = baseline_y - bar_height
        bar_right = bar_left + bar_width
        color = hex_to_rgb(palette[idx % len(palette)])
        draw.rectangle([bar_left, bar_top, bar_right, baseline_y], fill=color)
        draw.text((bar_left, baseline_y + 8), label, fill="black", font=font)
        draw.text((bar_left, bar_top - 16), f"{value:.4f}", fill="black", font=font)

    image.save(output_path)


def benchmark_model(name: str, model_factory, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> ModelBenchmark:
    cv_mean, cv_std = stratified_cv_score(model_factory, X_train, y_train, n_splits=5, random_state=42)

    model = model_factory()
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_classification(y_test, y_pred, y_proba)
    return ModelBenchmark(
        name=name,
        cv_accuracy_mean=cv_mean,
        cv_accuracy_std=cv_std,
        test_metrics=metrics,
        train_time_seconds=train_time,
    )


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    df = load_heart_dataset()
    bundle = prepare_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=0.2,
        random_state=42,
        stratify=bundle.y,
    )

    custom_benchmark = benchmark_model("custom", build_custom_model, X_train, y_train, X_test, y_test)
    sklearn_benchmark = benchmark_model("sklearn", build_sklearn_model, X_train, y_train, X_test, y_test)

    results = {
        "dataset": {
            "shape": list(bundle.X.shape),
            "target_name": bundle.target_name,
            "positive_share": float(bundle.y.mean()),
        },
        "custom": {
            "cv_accuracy_mean": custom_benchmark.cv_accuracy_mean,
            "cv_accuracy_std": custom_benchmark.cv_accuracy_std,
            "test_metrics": pretty_metrics(custom_benchmark.test_metrics),
            "train_time_seconds": custom_benchmark.train_time_seconds,
        },
        "sklearn": {
            "cv_accuracy_mean": sklearn_benchmark.cv_accuracy_mean,
            "cv_accuracy_std": sklearn_benchmark.cv_accuracy_std,
            "test_metrics": pretty_metrics(sklearn_benchmark.test_metrics),
            "train_time_seconds": sklearn_benchmark.train_time_seconds,
        },
    }
    (ARTIFACTS / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    save_bar_plot(
        ARTIFACTS / "accuracy_comparison.png",
        ["custom", "sklearn"],
        [custom_benchmark.test_metrics.accuracy, sklearn_benchmark.test_metrics.accuracy],
        "Test accuracy comparison",
        "Accuracy",
    )
    save_bar_plot(
        ARTIFACTS / "time_comparison.png",
        ["custom", "sklearn"],
        [custom_benchmark.train_time_seconds, sklearn_benchmark.train_time_seconds],
        "Training time comparison",
        "Seconds",
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
