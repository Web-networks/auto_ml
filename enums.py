from enum import Enum


class TaskType(Enum):
    Classification = 1
    Regression = 2


class DataType(Enum):
    Image = 1
    Text = 2
    Structured = 3


class BenchmarkDataset(Enum):
    MNIST = 1
    CIFAR10 = 2
    IMDB = 3


class Mode(Enum):
    Search = 1
    Optimize = 2
