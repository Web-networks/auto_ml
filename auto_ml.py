import logging
import time
import argparse
from enums import *
from context import Context
from states import get_initial_state


parser = argparse.ArgumentParser("auto_ml")
parser.add_argument('--mode', type=Mode, default='search', help='search -> 1\noptimize -> 2')
parser.add_argument('--data_type', type=DataType, default=DataType.Image,
                    help='image -> 1\ntext -> 2\nstructured data -> 3')
parser.add_argument('--task', type=TaskType, default=TaskType.Classification,
                    help='classification -> 1\nregression -> 2')
parser.add_argument('--data_file', type=BenchmarkDataset, default=BenchmarkDataset.mnist,
                    help='mnist -> 1\ncifar10 -> 2\nimdb -> 3')
parser.add_argument('--model_path', type=str, default='./model.h5', help="path to save model")
parser.add_argument("--base_arch_path", type=str, default=None, help="base architecture python file")
args = parser.parse_args()

logger = logging.getLogger("auto_ml")
fh = logging.FileHandler("auto_ml.log")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


def main():
    logger.info("Service start..")
    start = time.time()
    context = Context(args.task_type, args.data_type, args.data_file, args.mode, args.model_path, args.base_arch_path)
    state = get_initial_state()
    while state is not None:
        state.enter(context)
        state.handle(context)
        state = state.next()
    end = time.time()
    print(f"Overall time: {(end-start)/60}min.")
    logger.info("Service finished.")


if __name__ == '__main__':
    main()
