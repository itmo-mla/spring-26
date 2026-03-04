import pandas as pd
import PreparingDataset

#Гиперпараметр
min_gain_lvl = 0.01

def calculate_ginni(dicter):
    N = sum(dicter.values())
    summ = 0
    for val in dicter.values():
        summ += (val/N) ** 2
    return 1 - summ, N

def count_gain(parent_dicter, dicters):
    parent_ginni, N = calculate_ginni(parent_dicter)
    child_ginni = 0
    for dict in dicters.values():
        ginni, n = calculate_ginni(dict)
        child_ginni += ginni * (n / N)
    gain = parent_ginni - child_ginni
    return gain

def form_dicters(part, col_name, target):
    return {
        category: group.groupby(target)['w'].sum().to_dict()
        for category, group in part.groupby(col_name)
    }

def form_parent_dicter(part, col_name, target):
    return part.groupby(target)['w'].sum().to_dict()

def gain(part, col_name, target):
    dicters = form_dicters(part, col_name, target)
    dicter = form_parent_dicter(part, col_name, target)
    gain = count_gain(dicter, dicters)
    return gain

def choose_best_feature(df, target):
    cols = df.columns.to_list()
    cols.pop()
    cols.pop()
    if len(cols) == 0:
        return None
    max_gain = 0
    best_col = 0
    for col in cols:
        new_gain = gain(df, col, target)
        if new_gain > max_gain:
            max_gain = new_gain
            best_col = col
    if max_gain < min_gain_lvl:
        return None
    return best_col

def feature_decomposition(df, col_name):
    datasets = []
    nan_rows = []
    dicter = {}
    col = df[col_name]
    df = df.drop(columns=[col_name])
    for i in range(len(col)):
        row = df.iloc[i]
        group = col.iloc[i]
        if group is None or group == 'nan':
            nan_rows.append(row)
        else:
            if group in dicter:
                datasets[ dicter[group] ].append(row.to_dict())
            else:
                dicter[ group ] = len(dicter)
                datasets.append([row.to_dict()])

    weights = []
    for i in range(len(datasets)):
        summ = 0
        for j in range(len(datasets[i])):
            summ += datasets[i][j]['w']
        weights.append(summ)
    N = sum(weights)
    probs = []
    for elem in weights:
        probs.append(elem / N)

    for i in range(len(nan_rows)):
        raw = nan_rows[i]
        for j in range(len(probs)):
            row = raw.copy()
            row['w'] = raw['w'] * probs[j]
            datasets[j].append(row.to_dict())

    for i in range(len(datasets)):
        datasets[i] = pd.DataFrame(datasets[i])

    dicter2 = {} #для сохранения вероятностей
    for elem in dicter.keys():
        dicter2[elem] = probs[dicter[elem]]
        dicter[elem] = datasets[dicter[elem]]

    return dicter, dicter2

def decompose_dataset(df, target):
    feature = choose_best_feature(df, target)
    if feature is None:
        return None, None, None
    data, probs = feature_decomposition(df, feature)
    return data, feature, probs

class Node:
    def __init__(self):
        self.feature = None #признак, который будет правилом
        self.link = None #На родителя
        self.kids = {} #Пустрой словарь детей
        self.probs = {} #Пустой словарь вероятностей
        self.ans = None #Потенциальный ответ, для pruning сохраняем
        self.data = None #Только при pruning будет заполено данными
        self.correct = None  # Для pruning

class ListNode:
    def __init__(self):
        self.ans = None #Класс, который мы берем
        self.link = None #На родителя
        self.data = None #На этапе построения не нужна. Для pruning нужно!
        self.correct = None #Для pruning

#Придумать, как строить и хранить дерево, а так же придумать, как я буду это предсказывать!
def builder(df, target, parent_node):
    data, feature, probs = decompose_dataset(df, target)
    if data is None: #Лист по причине отсутствия признаков для разбиения или слаботы прироста информативности
        list_node = ListNode()
        list_node.link = parent_node
        list_node.ans = df[target].mode()[0]
        return list_node
    else:
        curr_node = Node()
        curr_node.link = parent_node
        curr_node.feature = feature
        curr_node.probs = probs
        curr_node.ans = df[target].mode()[0]
        for key in data.keys():
            curr_node.kids[key] = builder(data[key], target, curr_node)
        return curr_node

def predict_and_fill(df, now):
    if type(now) is Node:
        now.data = df
        for i in range(len(df)):
            if df[now.feature].iloc[i] not in now.kids.keys():
                df.loc[i, now.feature] = None

        df_none = df[df[now.feature].isna()]
        df_not_none = df[df[now.feature].notna()]

        data, _ = feature_decomposition(df_not_none, now.feature)
        for key in data.keys():
            data[key] = data[key].to_dict(orient='records')

        for i in range(len(df_none)):
            raw = df_none.iloc[i] #строка
            for key in now.probs.keys():
                row = raw.copy()
                row['w'] = now.probs[key] * raw['w']
                row = row.drop(now.feature)
                if key in data:
                    data[key].append(row.to_dict())
                else:
                    data[key] = [row.to_dict()]

        for key in data.keys():
            data[key] = pd.DataFrame(data[key])

        for key in data.keys():
            predict_and_fill(data[key], now.kids[key])

    else:
        now.data = df

def pruning_tree(now, target):
    if type(now) == ListNode:
        now.correct = 0
        for i in range(len(now.data)):
            if now.data.iloc[i][target] == now.ans:
                now.correct += now.data.iloc[i]['w']
        return now
    else:
        for key in now.kids.keys():
            now.kids[key] = pruning_tree(now.kids[key], target)

        my_count_correct = 0
        df = now.data
        for i in range(len(df)):
            if df.iloc[i][target] == now.ans:
                my_count_correct += df.iloc[i]['w']

        correct_nothing_summ = 0
        for key in now.kids.keys():
            correct_nothing_summ += now.kids[key].correct

        if my_count_correct >= correct_nothing_summ:
            new_list = ListNode()
            new_list.link = now.link
            new_list.data = now.data
            new_list.ans = now.ans
            new_list.correct = my_count_correct
            return new_list
        else:
            now.correct = correct_nothing_summ
            return now

def get_prediction(now, row):
    if type(now) == ListNode:
        return now.ans, row['w']
    else:
        if row[now.feature] in now.kids.keys():
            return get_prediction(now.kids[row[now.feature]], row)
        else:
            results = {}
            total_weight = 0

            for key, prob in now.probs.items():
                new_row = row.copy()
                new_row['w'] = row['w'] * prob

                ans, weight = get_prediction(now.kids[key], new_row)

                if ans in results:
                    results[ans] += weight
                else:
                    results[ans] = weight
                total_weight += weight

            if results:
                best_ans = max(results.items(), key=lambda x: x[1])[0]
                return best_ans, total_weight
            else:
                return now.ans, row['w']
