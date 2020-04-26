from enums import *


class Context:
    def __init__(self,
                 task_type: TaskType,
                 data_type: DataType,
                 data_file: str,
                 mode: Mode,
                 model_path: str = "model.h5",
                 base_model_path: str = None):
        self.task_type = task_type
        self.data_type = data_type
        self.data_file = data_file
        self.mode = mode
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.x_train = self.x_test = self.y_train = self.y_test = 0
        self.model = None
