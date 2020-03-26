import os
import sys
import logging
import autokeras as ak
import time
import argparse
from benchmark_data import load_data
from enums import TaskType, DataType, BenchmarkDataset

from auto_model_builder import GeneralModelBuilder, CustomModelBuilder

parser = argparse.ArgumentParser("auto_ml")
parser.add_argument('--mode', type=str, default='search', help='search or optimize')
parser.add_argument('--data_type', type=DataType, default=DataType.Image, help='image, text or structured data')
parser.add_argument('--train_test_split', type=float, default=0.3, help='test size')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--name', type=str, default='model_name', help='')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--task', type=TaskType, default=TaskType.Classification, help='classification or regression')
parser.add_argument('--time_limit', type=int, default=60, help='time limit in minutes')
parser.add_argument('--data', type=BenchmarkDataset, default=BenchmarkDataset.mnist, help='benchmark data (mnist, cifar10, imdb, titanic)')
parser.add_argument('--model_path', type=str, default='model.h5')
args = parser.parse_args()


def main():
    (x_train, y_train), (x_test, y_test) = load_data(args.data)
    if args.mode == 'search':
        builder = GeneralModelBuilder()
    else:
        builder = CustomModelBuilder()
    builder.set_name(args.name)
    builder.set_seed(args.seed)
    builder.set_data_type(args.data_type)
    builder.set_task(args.task)
    auto_model = builder.model
    auto_model.fit(x_train, y_train)
    auto_model.evaluate(x_test, y_test)
    model = auto_model.export_model()
    model.save(args.model_path)


if __name__ == '__main__':
    main()
