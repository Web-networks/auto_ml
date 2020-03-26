from enum import Enum


class TaskType(Enum):
    Classification = 1
    Regression = 2


class DataType(Enum):
    Image = 1
    Text = 2
    Structured = 3


class BenchmarkDataset(Enum):
    mnist = 1
    cifar10 = 2
    imdb = 3
    titanic = 4
