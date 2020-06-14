import keras
import numpy as np
import luigi
import autokeras as ak
from tensorflow.keras.models import load_model
from os.path import join as path_join
from tensorflow.keras.metrics import Accuracy, MeanSquaredError
from abc import ABC, abstractmethod, abstractproperty


class TrainData(luigi.Task):
    data_path = luigi.Parameter(description='path to data')

    def output(self):
        return luigi.LocalTarget(self.data_path)


class ValData(luigi.Task):
    data_path = luigi.Parameter(description='path to data')

    def output(self):
        return luigi.LocalTarget(self.data_path)


class TrainModel(ABC, luigi.Task):
    model_path = luigi.Parameter(description='path to directory where model will be located')
    data_type = luigi.Parameter(default='image', description='type of data (image, text or csv)')
    task_type = luigi.Parameter(default='classification', description='classification or regression')
    epochs = luigi.Parameter(default=10, description='number of epochs')
    data_path = luigi.Parameter(description='path to data')
    image_height = luigi.Parameter(default=28, description='image height')
    image_widht = luigi.Parameter(default=28, description='image widht')

    def requires(self):
        return TrainData(data_path=self.data_path)

    @abstractmethod
    def build_model(self):
        pass

    def run(self):

        auto_model = self.build_model()
        if auto_model is None:
            print('Error')
            return
        if self.data_type == 'csv':
            dataset = np.loadtxt(input().path)
            X = dataset[:, :-1]
            y = dataset[:, -1]
        else:
            if self.data_type == 'image':
                dataset = keras.preprocessing.image_dataset_from_directory(
                    input().path, batch_size=64, image_size=(self.image_height, self.image_widht))
            else:
                dataset = keras.preprocessing.text_dataset_from_directory(
                    input().path, batch_size=64)
            X = np.array()
            y = np.array()
            for data, labels in dataset:
                X = np.vstack((X, data))
                y = np.vstack((y, labels))
        auto_model.fit(X, y, epochs=self.epochs, time_limit=60 * 60)


        model = auto_model.export_model()
        print(type(model))
        try:
            model.save(self.output().path, save_format="tf")
        except:
            model.save(self.output().path)
        print(model.summary())

    def output(self):
        return luigi.LocalTarget(path_join(self.model_path, 'model'))


class SearchModel(TrainModel):
    def build_model(self) -> ak.AutoModel:
        model = None
        if self.data_type == 'image':
            if self.task_type == 'regression':
                model = ak.ImageRegressor()
            elif self.task_type == 'classification':
                model = ak.ImageClassifier()
        elif self.data_type == 'text':
            if self.task_type == 'regression':
                model = ak.TextRegressor()
            elif self.task_type == 'classification':
                model = ak.TextRegressor()
        elif self.data_type == 'csv':
            if self.task_type == 'regression':
                model = ak.StructuredDataRegressor()
            elif self.task_type == 'classification':
                model = ak.StructuredDataClassifier()
        return model


class OptimizeModel(TrainModel):
    def build_model(self):
        base_model_path = path_join(self.model_path, 'base_model.py')
        exec(open(base_model_path).read())
        model = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            max_trials=1)
        return model


class EvaluateModel(luigi.Task):
    data_path = luigi.Parameter(description='path to data')
    model_path = luigi.Parameter(description='path to directory where model will be located')
    metric = luigi.Parameter(default='accuracy', description='accuracy or mse')
    def requires(self):
        req = {
            'model' : TrainModel(),
            'train_data' : TrainData(),
            'val_data' : ValData(),
        }
        return req

    def run(self):
        model = load_model(input()['model'].path, custom_objects=ak.CUSTOM_OBJECTS)
        if self.data_type == 'csv':
            dataset = np.loadtxt(input().path)
            X = dataset[:, :-1]
            y = dataset[:, -1]
        else:
            if self.data_type == 'image':
                dataset = keras.preprocessing.image_dataset_from_directory(
                    input().path, batch_size=64, image_size=(self.image_height, self.image_widht))
            else:
                dataset = keras.preprocessing.text_dataset_from_directory(
                    input().path, batch_size=64)
            X = np.array()
            y = np.array()
            for data, labels in dataset:
                X = np.vstack((X, data))
                y = np.vstack((y, labels))
        prediction = model.predict(X)
        if self.metric == 'accuracy':
            metric = Accuracy()
        else:
            metric = MeanSquaredError()
        result = metric(y, prediction)
        print(f'{self.metric}:', result)
