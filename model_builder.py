from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
import autokeras as ak
from enums import *


class Director:
    def __init__(self, task_type: TaskType, data_type: DataType, base_model_path: str):
        self.task_type = task_type
        self.data_type = data_type
        self.base_model_path = base_model_path
        self.builder = Builder()

    def build_model(self) -> None:
        self.builder.produce_input_block(self.data_type)
        self.builder.produce_body(self.base_model_path)
        self.builder.produce_output_block(self.task_type)
        self.produce_model()


class Builder(ABC):
    def __init__(self):
        self.input_node = None
        self.auto_model = None
        self.output_node = None

    @abstractproperty
    def model(self) -> None:
        pass

    @abstractmethod
    def produce_model(self) -> None:
        pass

    def produce_input_block(self, data_type: DataType) -> None:
        if data_type == DataType.Image:
            self.input_node = ak.ImageInput()
        elif data_type == DataType.Text:
            self.input_node = ak.TextInput()
        elif data_type == DataType.Structured:
            self.input_node == ak.StructuredDataInput()

    @abstractmethod
    def produce_body(self, model_base_path: str) -> None:
        pass

    def produce_output_block(self, task_type: TaskType) -> None:
        if task_type == TaskType.Regression:
            self.output_node = ak.RegressionHead()
        elif task_type == TaskType.Classification:
            self.output_node = ak.ClassificationHead()


class SearchBuilder(Builder):
    def produce_model(self) -> None:
        self.output_node = self.output_node(self.input_node)
        self.auto_model = ak.AutoModel(inputs=self.input_node, outputs=self.output_node)

    def produce_body(self, model_base_path: str) -> None:
        pass

    @property
    def model(self) -> None:
        return self.auto_model


class OptimizeBuilder(Builder):
    def __init__(self):
        super(OptimizeBuilder, self).__init__()
        self.body = None

    def produce_model(self) -> None:
        self.body = self.body(self.input_node)
        self.output_node = self.output_node(self.body)
        self.auto_model = ak.AutoModel(inputs=self.input_node, outputs=self.output_node)

    def produce_body(self, model_base_path: str) -> None:
        exec(open(model_base_path).read())
        self.body = CustomBlock()

    @property
    def model(self) -> None:
        return self.auto_model
