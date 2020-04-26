from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
import logging
from data_loader import load_data
from utils import *
from model_builder import *
from enums import *

logger = logging.getLogger("auto_ml")
fh = logging.FileHandler("auto_ml.log")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


class State(ABC):
    @abstractmethod
    def enter(self, context) -> None:
        pass
    @abstractmethod
    def handle(self, context) -> None:
        pass
    @abstractmethod
    def next(self) -> State:
        pass


class LoadDataState(State):
    def enter(self, context) -> None:
        logger.info("Loading data..")

    def handle(self, context) -> None:
        (x_train, y_train), (x_test, y_test) = load_data(context.data_file)
        context.x_train = x_train
        context.y_train = y_train
        context.x_test = x_test
        context.y_test = y_test

    def next(self) -> State:
        return BuildModelState()


class BuildModelState(State):
    def enter(self, context) -> None:
        logger.info("Building model..")

    def handle(self, context) -> None:
        director = Director(context.task_type,
                            context.data_type,
                            context.base_model_path)
        if context.mode == Mode.Search:
            builder = SearchBuilder()
        else:
            builder = OptimizeBuilder()
        director.builder = builder
        director.build_model()
        context.model = builder.model

    def next(self) -> State:
        return TrainModelState()


class TrainModelState(State):
    def enter(self, context) -> None:
        logger.info("Training model..")

    def handle(self, context) -> None:
        train(context.model, context.x_train, context.y_train)

    def next(self) -> State:
        return EvalModelState()


class EvalModelState(State):
    def enter(self, context) -> None:
        logger.info("Evaluating model..")

    def handle(self, context) -> None:
        print(context.model.summary())
        score = evaluate(context.model, context.x_test, context.y_test)
        print(f"Eval: {score}")

    def next(self) -> State:
        return SaveModelState()


class SaveModelState(State):
    def enter(self, context) -> None:
        logger.info("Saving model..")

    def handle(self, context) -> None:
        save(context.model, context.model_path)

    def next(self) -> State:
        return None


def get_initial_state() -> State:
    return LoadDataState()
