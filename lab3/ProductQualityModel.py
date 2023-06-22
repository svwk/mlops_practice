import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import part_params
from MessageConstants import MessageConstants


class ProductQualityModel:
    target_column_name = "z"

    def __init__(self,
                 model_file_path: str = None,
                 scaler_file_path: str = None):

        self.mc = MessageConstants()

        self.model_file_path = model_file_path
        self.scaler_file_path = scaler_file_path

        self.model = None
        self.scaler = None


    def load_model(self):
        """
        Загрузка моделей масштабатора и линейной регрессии
        """
        if self.model_file_path is not None:
            try:
                self.model = joblib.load(self.model_file_path)
            except Exception as inst:
                print(f"{self.mc.MODEL_READ_ERROR} {self.model_file_path} {inst.args}")

        if self.scaler_file_path is not None:
            try:
                self.scaler = joblib.load(self.scaler_file_path)

            except Exception as inst:
                print(f"{self.mc.MODEL_READ_ERROR} {self.scaler_file_path} {inst.args}")

        print("Model loaded")


    def save_model(self):
        """
        Сохранение моделей масштабатора и линейной регрессии
        """
        if self.model_file_path is not None:
            try:
                file_list = joblib.dump(self.model, self.model_file_path)
                if len(file_list) > 0:
                    print(f"{self.mc.MODEL_SAVED}: {self.model_file_path}")
                else:
                    print(f"{self.mc.FILE_SAVE_ERROR} {self.model_file_path}")
            except Exception as inst:
                print(f"{self.mc.FILE_SAVE_ERROR} {self.model_file_path}: {inst.args}")

        if self.scaler_file_path is not None:
            try:
                file_list = joblib.dump(self.scaler, self.scaler_file_path)
                if len(file_list) > 0:
                    print(f"{self.mc.MODEL_SAVED}: {self.scaler_file_path}")
                else:
                    print(f"{self.mc.FILE_SAVE_ERROR} {self.scaler_file_path}")
            except Exception as inst:
                print(f"{self.mc.FILE_SAVE_ERROR} {self.scaler_file_path}: {inst.args}")


    def clean_data(self, raw_dataset):
        """
        Предобработка данных. Удаление пустых строк и дубликатов
        :param raw_dataset:  Исходный датасет
        """

        pre_count = raw_dataset.shape[0]
        dataset = raw_dataset.drop_duplicates()
        dataset = dataset.drop(dataset[(dataset.l1.isnull()) | (dataset.l1 == 0)].index)
        dataset = dataset.drop(dataset[(dataset.l2.isnull()) | (dataset.l2 == 0)].index)
        dataset = dataset.drop(dataset[
                                   (dataset.status.isnull()) |
                                   (~dataset.status.isin([part_params.godn_name, part_params.brak_name]))
                                   ].index)

        dataset = dataset.reset_index(drop=True)

        post_count = dataset.shape[0]
        print(f"{self.mc.STRING_COUNT_PRE}: {pre_count}")
        print(f"{self.mc.STRING_COUNT_POST}: {post_count}")

        return dataset


    def get_size_deviations(self, raw_dataset):
        """
        Вычисление  отклонений фактических размеров относительно номинальных
        :param raw_dataset: Исходный датасет
        :return: отклонения размеров
        """
        x = (raw_dataset['l2'] - part_params.l2_nom).values
        y = (raw_dataset['l1'] - part_params.l1_nom).values

        return np.array([x, y]).reshape(-1, 2)


    def fit_scaler(self, raw_dataset=None, size_deviations=None):
        """
        Вычисление параметров масштабатора
        :param raw_dataset: Исходный датасет
        :param size_deviations:  отклонения размеров
        :return: отклонения размеров
        """
        if size_deviations is None and raw_dataset is None:
            return None

        data = size_deviations if size_deviations is not None else \
            self.get_size_deviations(raw_dataset)
        self.scaler.fit(data)

        return data


    def transform_xy(self, raw_dataset=None, size_deviations=None):
        """
        Предобработка данных
        :param size_deviations:  отклонения размеров
        :param raw_dataset:  Исходный датасет
        """

        if size_deviations is None and raw_dataset is None:
            return None

        data = size_deviations if size_deviations is not None else \
            self.get_size_deviations(raw_dataset)

        data = self.scaler.transform(data)
        data = self.to_polynom(data, order=3)

        num_columns = ['x', 'y', 'x2', 'y2', 'x3', 'y3']
        df_standard = pd.DataFrame(data, columns=num_columns)

        return df_standard


    def to_polynom(self, x, order=2):
        """
        Convert to polynomial
        :param x: initial data
        :param order: polynomial degree
        """
        order_range = range(2, order + 1, 1)
        out = np.copy(x)
        for i in order_range:
            out = np.hstack([out, np.power(x, i)])
        return out


    def get_z_value(self, status_value):
        """
        Вычисление столбца z
        :param status_value: значение столбца 'status'
        """
        return part_params.godn if status_value == part_params.godn_name else part_params.brak


    def get_status_value(self, z_value):
        """
        Вычисление столбца 'status'
        :param status_value: значение столбца z
        """
        return part_params.godn_name if z_value == part_params.godn else part_params.brak_name


    def add_z_feature(self, raw_dataset, target_dataset):
        """
        Добавление к датасету столбца z
        :param raw_dataset: Исходный датасет
        :param target_dataset: Целевой датасет
        """
        target_dataset['z'] = raw_dataset['status'].apply(lambda x: self.get_z_value(x))
        return target_dataset


    def preprocess_input_data(self, raw_dataset, to_fit_scaler=False):
        """
        Полная предобработка данных
        :param raw_dataset: Исходный датасет
        :param to_fit_scaler: Флаг нужно ли вычислять масштабатор
        :return:
        """
        clean_dataset = self.clean_data(raw_dataset)

        data = self.fit_scaler(clean_dataset) if to_fit_scaler \
            else self.get_size_deviations(clean_dataset)

        processed_dataset = self.transform_xy(size_deviations=data)
        self.add_z_feature(clean_dataset, processed_dataset)

        return processed_dataset


    def separate_dataset(self, raw_dataset, random_state=42, train_filename=None, test_filename=None):
        """
         Разделение на обучающую и тестовую выборки
        :param raw_dataset: Исходный датасет
        :param random_state: фиксированный сид случайных чисел (для повторяемости)
        :return: Два дата-фрейма с обучающими и тесовыми данными
        """

        column = ProductQualityModel.target_column_name
        z = raw_dataset[column]  # .values
        xy = raw_dataset.drop(column, axis=1)

        xy_train, xy_test, z_train, z_test = train_test_split(xy, z, test_size=0.3,
                                                              random_state=random_state)
        xy_train[column] = z_train
        xy_test[column] = z_test

        if train_filename is not None:
            try:
                xy_train.to_csv(train_filename, index=False)
                print(f"{self.mc.SOURCE_SAVED}: {train_filename}")
            except Exception as ex:
                print(f"{self.mc.FILE_SAVE_ERROR} {train_filename}: {ex.args}")

        if test_filename is not None:
            try:
                xy_test.to_csv(test_filename, index=False)
                print(f"{self.mc.SOURCE_SAVED}: {test_filename}")
            except Exception as ex:
                print(f"{self.mc.FILE_SAVE_ERROR} {test_filename}: {ex.args}")

        return xy_train, xy_test


    def train_model(self, train_dataset):
        """
        Обучение модели
        :param train_dataset:  Обучающий набор данных
        """

        column = ProductQualityModel.target_column_name
        y_train = train_dataset[column].values
        x_train = train_dataset.drop(column, axis=1).values

        self.model = LogisticRegression(max_iter=100_000).fit(x_train, y_train)

        return self.model


    def test_model(self, test_dataset):
        """
        Проверка качества модели
        :param test_dataset:  Тестовый набор данных
        """
        column = ProductQualityModel.target_column_name
        y_test = test_dataset[column].values
        xy_test = test_dataset.drop(column, axis=1).values

        accuracy = self.model.score(xy_test, y_test)
        print(f'{self.mc.MODEL_TEST_ACCURACY}: {accuracy:.3f}')

        return accuracy


    def compute_model_from_raw(self, raw_dataset):
        """
        Вычисление модели по сырым данным
        :param raw_dataset: Исходный датасет
        """
        processed_dataset = self.preprocess_input_data(raw_dataset, True)
        return self.compute_model_from_processed(processed_dataset)


    def compute_model_from_processed(self, processed_dataset):
        """
        Вычисление модели по предобработанным данным
        :param processed_dataset: Предобработанный датасет
        """
        xy_train, xy_test = self.separate_dataset(processed_dataset)
        self.train_model(xy_train)
        self.test_model(xy_test)
        self.save_model()
        return True


    def predict(self, part_length: float, part_width: float):
        """Part quality conformity prediction using user input
        - **part_length**: part length
        - **part_width**: part width
        """
        if part_params.l1_nom + 1 < part_length or part_length < part_params.l1_nom - 1 or \
                part_params.l2_nom + 1 < part_width or part_width < part_params.l2_nom - 1:
            return self.mc.INVALID_MEASUREMENTS
        try:
            if (self.model is not None) and (self.scaler is not None):
                df = pd.DataFrame({'l1': [part_length], 'l2': [part_width]})
                df = self.transform_xy(df)
                z_predict = self.model.predict(df.values)[0]
                return self.get_status_value(z_predict)
            else:
                return self.mc.MODEL_READ_ERROR
        except Exception as e:
            return f"{self.mc.PARAMETERS_NOT_VALID}. {e}."
