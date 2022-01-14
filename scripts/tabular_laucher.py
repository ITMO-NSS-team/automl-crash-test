import os
from scripts.utils import path_to_save_results


class TabularLauncher:
    def __init__(self,
                 data: tuple,
                 task: str = 'classicfication',
                 params: dict = None,
                 dataset_name: str = None,
                 framework_name: str = None,
                 launch: int = None,
                 logger: object = None,
                 helper: object = None,
                 ):
        self.train_data = data[0]
        self.test_data = data[1]
        self.params = params
        self.dataset_name = dataset_name
        self.launch = launch
        self.logger = logger
        self.metrics = None
        self.helper = helper
        self.task = task

    def perform_experiment(self):
        return

    def fit(self):
        return

    def predict(self):
        return
