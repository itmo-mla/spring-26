from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


ITEMS_PATH = "data/items.csv"
INTERACTIONS_PATH = "data/interactions.csv"
USERS_PATH = "data/users.csv"

TOPICS = {
    "machine_learning": [
        ("Введение в деревья решений", "деревья решений классификация признаки энтропия информация обучение модель"),
        ("Градиентный бустинг на табличных данных", "градиентный бустинг ансамбль деревья loss функция качество табличные данные"),
        ("Регуляризация линейных моделей", "линейная регрессия логистическая регрессия регуляризация l1 l2 переобучение"),
        ("Метрики качества классификации", "accuracy precision recall f1 roc auc confusion matrix оценка модели"),
        ("Кросс-валидация и подбор параметров", "cross validation grid search train test split гиперпараметры качество"),
        ("Ансамблевые методы", "bagging boosting random forest stacking ансамбль моделей устойчивость качество"),
    ],
    "natural_language": [
        ("TF-IDF для анализа текстов", "текст корпус токенизация tfidf векторизация слова документы признаки"),
        ("Латентно-семантический анализ", "lsa svd латентные факторы семантика текст документы матрица терм"),
        ("Классификация новостей", "новости текст классификация наивный байес логистическая регрессия признаки"),
        ("Поиск похожих документов", "косинусное сходство документы embedding поиск похожий текст рекомендация"),
        ("Предобработка русского текста", "лемматизация стоп слова нормализация токены морфология русский язык"),
        ("Тематическое моделирование", "topic modeling lda темы документы распределение слов вероятностная модель"),
    ],
    "databases": [
        ("Индексы в реляционных базах", "база данных sql индекс btree запрос оптимизация таблица"),
        ("Нормализация схем данных", "нормализация база данных ключ связь таблицы формы зависимости"),
        ("Транзакции и ACID", "транзакции acid изоляция атомарность согласованность durability база данных"),
        ("Аналитические хранилища", "data warehouse olap витрина данных аналитика sql агрегация"),
        ("NoSQL модели хранения", "nosql key value document graph column database масштабирование"),
        ("Оптимизация SQL-запросов", "sql explain join index query optimizer performance база данных"),
    ],
    "computer_vision": [
        ("Свёрточные нейронные сети", "computer vision cnn convolution pooling image classification нейронная сеть"),
        ("Детекция объектов", "object detection bounding box yolo rcnn изображение компьютерное зрение"),
        ("Сегментация изображений", "segmentation mask image pixel unet semantic instance vision"),
        ("Аугментация изображений", "augmentation image rotate crop flip noise dataset обучение"),
        ("Распознавание лиц", "face recognition embedding similarity verification computer vision"),
        ("Классификация изображений", "image classification resnet efficientnet cnn dataset accuracy"),
    ],
    "security": [
        ("Основы криптографии", "cryptography encryption key hash signature security protocol"),
        ("Анализ сетевых атак", "network attack intrusion detection traffic anomaly security"),
        ("Безопасность веб-приложений", "web security xss csrf sql injection vulnerability authentication"),
        ("Управление доступом", "access control role permission authentication authorization security"),
        ("Логи и мониторинг безопасности", "logs monitoring siem anomaly detection incident security"),
        ("Защита персональных данных", "privacy personal data gdpr anonymization security compliance"),
    ],
    "recommenders": [
        ("Коллаборативная фильтрация", "recommendation collaborative filtering user item rating matrix similarity"),
        ("Матричная факторизация", "matrix factorization latent factors recommender svd rating prediction"),
        ("SLIM-рекомендатель", "slim sparse linear method item item model recommender l1 regularization"),
        ("Оценка рекомендательных систем", "rmse ndcg precision recall recommender ranking metrics"),
        ("Content-based рекомендации", "content based recommendation tfidf text profile similarity items"),
        ("Гибридные рекомендатели", "hybrid recommender collaborative content model ranking personalization"),
    ],
}

USER_PROFILES = [
    ("u01", ["machine_learning", "recommenders"]),
    ("u02", ["natural_language", "machine_learning"]),
    ("u03", ["databases", "security"]),
    ("u04", ["computer_vision", "machine_learning"]),
    ("u05", ["security", "databases"]),
    ("u06", ["recommenders", "natural_language"]),
    ("u07", ["databases", "machine_learning"]),
    ("u08", ["computer_vision", "natural_language"]),
    ("u09", ["security", "recommenders"]),
    ("u10", ["natural_language", "databases"]),
    ("u11", ["machine_learning", "security"]),
    ("u12", ["recommenders", "computer_vision"]),
    ("u13", ["natural_language", "recommenders"]),
    ("u14", ["databases", "recommenders"]),
    ("u15", ["computer_vision", "security"]),
    ("u16", ["machine_learning", "natural_language"]),
    ("u17", ["security", "machine_learning"]),
    ("u18", ["computer_vision", "recommenders"]),
    ("u19", ["databases", "natural_language"]),
    ("u20", ["recommenders", "security"]),
    ("u21", ["machine_learning", "databases"]),
    ("u22", ["natural_language", "security"]),
    ("u23", ["computer_vision", "databases"]),
    ("u24", ["recommenders", "machine_learning"]),
]


def _create_items() -> pd.DataFrame:
    rows = []
    item_number = 1
    for topic, documents in TOPICS.items():
        for local_index, (title, text) in enumerate(documents, start=1):
            rows.append({
                "item_id": f"i{item_number:02d}",
                "topic": topic,
                "title": title,
                "text": text,
            })
            item_number += 1
    return pd.DataFrame(rows)


def _create_users() -> pd.DataFrame:
    return pd.DataFrame([
        {"user_id": user_id, "primary_topic": topics[0], "secondary_topic": topics[1]}
        for user_id, topics in USER_PROFILES
    ])


def _create_interactions(items: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    rows = []
    topic_to_items = {
        topic: items.loc[items["topic"] == topic, "item_id"].tolist()
        for topic in TOPICS
    }

    for user_id, preferred_topics in USER_PROFILES:
        selected_items = []
        selected_items.extend(rng.choice(topic_to_items[preferred_topics[0]], size=5, replace=False))
        selected_items.extend(rng.choice(topic_to_items[preferred_topics[1]], size=4, replace=False))

        other_topics = [topic for topic in TOPICS if topic not in preferred_topics]
        for topic in other_topics:
            selected_items.extend(rng.choice(topic_to_items[topic], size=2, replace=False))

        selected_items = list(dict.fromkeys(selected_items))

        for item_id in selected_items:
            item_topic = items.loc[items["item_id"] == item_id, "topic"].iloc[0]
            if item_topic == preferred_topics[0]:
                base = 4.55
            elif item_topic == preferred_topics[1]:
                base = 4.05
            else:
                base = 2.35

            rating = base + rng.normal(0.0, 0.35)
            rating = float(np.clip(np.round(rating, 1), 1.0, 5.0))
            rows.append({"user_id": user_id, "item_id": item_id, "rating": rating})

    return pd.DataFrame(rows)


def ensure_dataset(
    items_path: str = ITEMS_PATH,
    interactions_path: str = INTERACTIONS_PATH,
    users_path: str = USERS_PATH,
) -> Tuple[Path, Path, Path]:
    items_file = Path(items_path)
    interactions_file = Path(interactions_path)
    users_file = Path(users_path)

    if items_file.exists() and interactions_file.exists() and users_file.exists():
        return items_file, interactions_file, users_file

    items_file.parent.mkdir(parents=True, exist_ok=True)
    items = _create_items()
    users = _create_users()
    interactions = _create_interactions(items)

    items.to_csv(items_file, index=False)
    interactions.to_csv(interactions_file, index=False)
    users.to_csv(users_file, index=False)
    return items_file, interactions_file, users_file


def split_interactions_by_user(interactions: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    train_parts = []
    test_parts = []

    for _, group in interactions.groupby("user_id"):
        group = group.sample(frac=1.0, random_state=int(rng.randint(0, 10**9))).reset_index(drop=True)
        n_test = max(1, int(round(len(group) * test_ratio)))
        test_parts.append(group.iloc[:n_test])
        train_parts.append(group.iloc[n_test:])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, test


def make_mappings(users: pd.DataFrame, items: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    user_to_index = {user_id: index for index, user_id in enumerate(users["user_id"].tolist())}
    item_to_index = {item_id: index for index, item_id in enumerate(items["item_id"].tolist())}
    return user_to_index, item_to_index


def build_user_item_matrix(
    interactions: pd.DataFrame,
    n_users: int,
    n_items: int,
    user_to_index: Dict[str, int],
    item_to_index: Dict[str, int],
) -> np.ndarray:
    matrix = np.zeros((n_users, n_items), dtype=float)
    for row in interactions.itertuples(index=False):
        user_index = user_to_index[row.user_id]
        item_index = item_to_index[row.item_id]
        matrix[user_index, item_index] = float(row.rating)
    return matrix


def add_indices(interactions: pd.DataFrame, user_to_index: Dict[str, int], item_to_index: Dict[str, int]) -> pd.DataFrame:
    indexed = interactions.copy()
    indexed["user_index"] = indexed["user_id"].map(user_to_index)
    indexed["item_index"] = indexed["item_id"].map(item_to_index)
    return indexed


def load_text_recommendation_dataset():
    items_path, interactions_path, users_path = ensure_dataset()
    items = pd.read_csv(items_path)
    interactions = pd.read_csv(interactions_path)
    users = pd.read_csv(users_path)

    user_to_index, item_to_index = make_mappings(users, items)
    train_interactions, test_interactions = split_interactions_by_user(interactions)

    train_interactions = add_indices(train_interactions, user_to_index, item_to_index)
    test_interactions = add_indices(test_interactions, user_to_index, item_to_index)

    train_matrix = build_user_item_matrix(
        train_interactions,
        n_users=len(users),
        n_items=len(items),
        user_to_index=user_to_index,
        item_to_index=item_to_index,
    )

    return {
        "items": items,
        "users": users,
        "interactions": interactions,
        "train_interactions": train_interactions,
        "test_interactions": test_interactions,
        "train_matrix": train_matrix,
        "n_users": len(users),
        "n_items": len(items),
        "user_to_index": user_to_index,
        "item_to_index": item_to_index,
    }
