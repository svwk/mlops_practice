"""
Методы работы с данными
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import message_constants as mc
from detail_params import l2_nom, l1_nom, godn, brak, godn_name, brak_name


def print_error(ex, file_path):
    """
    Вывод сообщения об ошибке сохранения файла
    :param ex: объект исключения
    :param file_path:  полный путь файла
    """
    print(f"{mc.FILE_SAVE_ERROR} {file_path}: {ex.args}")


def separate_dataset(source_dataset, random_state=42):
    """
     Разделение на обучающую и тестовую выборки
    :param source_dataset: Исходный датасет
    :param random_state: фиксированный сид случайных чисел (для повторяемости)
    :return: Два дата-фрейма с обучающими и тесовыми данными
    """

    target_column_name = "z"
    z = source_dataset[target_column_name]  # .values
    xy = source_dataset.drop(target_column_name, axis=1)

    xy_train, xy_test, z_train, z_test = train_test_split(xy, z, test_size=0.3,
                                                          random_state=random_state)
    xy_train[target_column_name] = z_train
    xy_test[target_column_name] = z_test

    return xy_train, xy_test


def preprocess_data(source_dataset, scaler, to_fit_scaler):
    """
    Предобработка данных
    :param source_dataset:  Исходный датасет
    :param scaler: объект StandardScaler, выполняющий стандартизацию
    :param to_fit_scaler: флаг нужно ли обучать об]ект scaler
    """

    pre_count = source_dataset.shape[0]
    dataset = source_dataset.drop_duplicates()
    dataset = dataset.drop(dataset[(dataset.l1.isnull()) | (dataset.l1 == 0)].index)
    dataset = dataset.drop(dataset[(dataset.l2.isnull()) | (dataset.l2 == 0)].index)
    dataset = dataset.drop(dataset[(dataset.status.isnull()) | (~dataset.status.isin([godn_name, brak_name]))].index)

    dataset = dataset.reset_index(drop=True)

    dataset['x'] = dataset['l2'] - l2_nom
    dataset['y'] = dataset['l1'] - l1_nom

    num_columns = ['x', 'y']

    if to_fit_scaler:
        scaler.fit(dataset[num_columns])

    scaled = scaler.transform(dataset[num_columns])
    scaled = to_polynom(scaled, order=3)

    num_columns = np.hstack([num_columns, ['x2', 'y2', 'x3', 'y3']])

    df_standard = pd.DataFrame(scaled, columns=num_columns)
    df_standard['z'] = dataset['status'].apply(lambda x: godn if x == godn_name else brak)

    post_count = dataset.shape[0]
    print(f"{mc.STRING_COUNT_PRE}: {pre_count}")
    print(f"{mc.STRING_COUNT_POST}: {post_count}")

    return df_standard


# %% Загрузим, обучим и сохраним модель
def train_model(train_dataset):
    """
    Обучение модели
    :param train_dataset:  Обучающий набор данных
    """

    target_column_name = "z"
    y_train = train_dataset[target_column_name].values
    x_train = train_dataset.drop(target_column_name, axis=1).values

    model = LogisticRegression(max_iter=100_000).fit(x_train, y_train)

    return model


def test_model(test_dataset, model):
    """
    Проверка модели
    :param test_dataset:  Тестовый набор данных
    :param model: Проверяемая модель
    """
    target_column_name = "z"
    y_test = test_dataset[target_column_name].values
    xy_test = test_dataset.drop(target_column_name, axis=1).values

    accuracy = model.score(xy_test, y_test)
    print(f'{mc.MODEL_TEST_ACCURACY}: {accuracy:.3f}')

    return accuracy


def to_polynom(x, order=2):
    """
    Преобразование к полиному
    :param x: исходные данные
    :param order: степень полинома
    """
    order_range = range(2, order + 1, 1)
    out = np.copy(x)
    for i in order_range:
        out = np.hstack([out, np.power(x, i)])
    return out
