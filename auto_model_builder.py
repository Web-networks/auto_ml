from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
import time
import autokeras as ak
from enums import TaskType, DataType


class AbstractModelBuilder(ABC):
    @abstractproperty
    def model(self) -> None:
        pass

    @abstractmethod
    def set_task(self, task) -> None:
        pass

    @abstractmethod
    def set_data_type(self, data_type) -> None:
        pass

    @abstractmethod
    def add_block(self, block) -> None:
        pass

    def set_seed(self, seed) -> None:
        self._seed = seed

    def set_objective(self, objective) -> None:
        self._objective = objective

    def set_name(self, name) -> None:
        self._name = name

    def set_max_trials(self, max_trials) -> None:
        self._max_trials = max_trials


class GeneralModelBuilder(AbstractModelBuilder):
    def add_block(self, block) -> None:
        pass

    def __init__(self):
        self.reset()

    def reset(self):
        self._data_type = None
        self._task = None
        self._max_trials = 1
        self._objective = 'val_loss'
        self._name = 'model' + str(time.time())
        self._seed = 42

    @property
    def model(self) -> ak.AutoModel:
        model = None
        if self._task == TaskType.Classification:
            if self._data_type == DataType.Image:
                model = ak.ImageClassifier(max_trials=self._max_trials, objective=self._objective,
                                           name=self._name, seed=self._seed)
            elif self._data_type == DataType.Text:
                model = ak.TextClassifier(max_trials=self._max_trials, objective=self._objective,
                                          name=self._name, seed=self._seed)
            elif self._data_type == DataType.Structured:
                model = ak.StructuredDataClassifier(max_trials=self._max_trials, objective=self._objective,
                                                    name=self._name, seed=self._seed)
        elif self._task == TaskType.Regression:
            if self._data_type == DataType.Image:
                model = ak.ImageRegressor(max_trials=self._max_trials, objective=self._objective,
                                          name=self._name, seed=self._seed)
            elif self._data_type == DataType.Text:
                model = ak.TextRegressor(max_trials=self._max_trials, objective=self._objective,
                                         name=self._name, seed=self._seed)
            elif self._data_type == DataType.Structured:
                model = ak.StructuredDataRegressor(max_trials=self._max_trials, objective=self._objective,
                                                   name=self._name, seed=self._seed)
        self.reset()
        return model

    def set_data_type(self, data_type) -> None:
        self._data_type = data_type

    def set_task(self, task) -> None:
        self._task = task


class CustomModelBuilder(AbstractModelBuilder):
    def __init__(self) -> None:
        self.reset()

    @property
    def model(self) -> ak.AutoModel:
        if self._input_node is None or self._output_node is None:
            raise RuntimeError()
        model = ak.AutoModel(inputs=self._input_node,
                             outputs=self._output_node,
                             name=self._name,
                             objective=self._objective,
                             seed=self._seed)
        return model

    def reset(self):
        self._input_node = None
        self._output_node = None
        self._has_layer = False
        self._data_type = None
        self._task = None
        self._max_trials = 1
        self._objective = 'val_loss'
        self._name = 'model' + str(time.time())
        self._seed = 42

    def set_task(self, task) -> None:
        output_node = None
        if self._input_node is None:
            raise RuntimeError()
        self._task = task
        if task == TaskType.Classification:
            output_node = ak.ClassificationHead()
        elif task == TaskType.Regression:
            output_node = ak.RegressionHead()
        if output_node is None:
            raise RuntimeError()
        if self._has_layer:
            self._output_node = output_node(self._output_node)
        else:
            self._output_node = output_node(self._input_node)

    def set_data_type(self, data_type) -> None:
        self._data_type = data_type
        if data_type == DataType.Image:
            self._input_node = ak.ImageInput()
        elif data_type == DataType.Text:
            self._input_node = ak.TextInput()
        elif data_type == DataType.Structured:
            self._input_node = ak.StructuredDataInput()

    def add_block(self, block) -> None:
        if self._input_node is None:
            raise RuntimeError()
        if self._has_layer:
            self._output_node = block(self._output_node)
        else:
            self._output_node = block(self._input_node)

