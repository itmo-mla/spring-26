# Лабораторная работа №5. Рекомендательные системы

В работе реализуются SLIM и латентная модель для рекомендательных систем. SLIM сравнивается с эталонной реализацией `KarypisLab/SLIM`, латентная модель сравнивается с `TruncatedSVD`.

## Датасет

Используется Hugging Face датасет [`mstz/speeddating`](https://huggingface.co/datasets/mstz/speeddating), основанный на speed dating experiment.

В качестве пользователей используются признаки `dater`, в качестве объектов рекомендации — признаки `dated`, оценка строится по полю `is_match`.

## Реализация

- `source/model/slim.py`
- `source/model/lsm.py`
- `source/model/baselines.py`

## Оценка

- `RMSE`
- `NDCG@10`

Сравниваются собственный SLIM с эталонной реализацией [`KarypisLab/SLIM`](https://github.com/KarypisLab/SLIM) и собственная латентная модель с эталонной моделью на `TruncatedSVD`.

Собственная LSM реализуется через разные латентные факторы для пользователей и объектов. Для пользователя `u` обучается вектор `p_u`, для объекта `i` обучается вектор `q_i`, а предсказание считается как скалярное произведение:

Эталонная латентная модель использует другое представление: `TruncatedSVD` восстанавливает матрицу взаимодействий через усеченное сингулярное разложение. (Не нашел на просторах инета нормальную реализацию)

Для запуска `karypis_slim_reference` нужно отдельно собрать и установить Python-пакет из `KarypisLab/SLIM`.

## Установка KarypisLab/SLIM

Клонировать репозиторий:

```bash
cd spring-26/shinyakov-ad
git clone --recursive https://github.com/KarypisLab/SLIM.git
cd SLIM
```

Исправить в ```SLIM/CMakeLists.txt``` и ```SLIM/lib/GKlib```:

```
cmake_minimum_required(VERSION 2.8) -> cmake_minimum_required(VERSION 3.5)
```

Собрать `GKlib`:

```bash
cd lib/GKlib
make config openmp=set
make
cd ../..
```

Собрать и установить SLIM:

```bash
make config shared=1 cc=gcc cxx=gcc prefix=$HOME/.local
make install
```

Установить Python-пакет:

```bash
cd python-package
python setup.py install --user
```

Проверить установку:

```bash
python -c "from SLIM import SLIM, SLIMatrix; print('ok')"
```

Если на macOS возникают ошибки с `gcc` или OpenMP, установить компилятор через Homebrew:

```bash
brew install gcc cmake
```

После этого собрать SLIM с Homebrew GCC:

```bash
cd /Users/ovoshchko/ITMO-PiRSII/spring-26/SLIM
make config shared=1 cc=gcc-15 cxx=g++-15 prefix=$HOME/.local
make install
cd python-package
python setup.py install --user
```

## Запуск

```bash
python source/main.py
```

Графики сохраняются в `artifacts`.
