from tensorflow.keras.datasets import mnist, cifar10, imdb
from enums import BenchmarkDataset


def load_data(data):
    if data == BenchmarkDataset.mnist:
        return mnist.load_data()
    elif data == BenchmarkDataset.cifar10:
        return cifar10.load_data()
    elif data == BenchmarkDataset.imdb:
        return imdb.load_data()
