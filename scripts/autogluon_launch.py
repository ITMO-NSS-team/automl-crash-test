import timeit
import os
from autogluon.tabular import TabularPredictor


def path_to_save_results() -> str:
    path = project_path()
    save_path = os.path.join(path, 'results_of_experiments')
    return save_path


def project_path() -> str:
    name_project = 'automl-crash-test'
    abs_path = os.path.abspath(os.path.curdir)
    while os.path.basename(abs_path) != name_project:
        abs_path = os.path.dirname(abs_path)
    return abs_path


def exception_decorator(function_to_decorate):
    def exception_wrapper():
        try:
            function_to_decorate()
        except Exception:
            return None

        return exception_wrapper


class TabularLauncher:
    # TODO refactor
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


class AutoGluonRun(TabularLauncher):
    # TODO refactor
    def __init__(self,
                 data: tuple,
                 task: str = 'classicfication'):
        super().__init__(data, task)
        self.task_type = task

    def fit(self):

        last_name = self.train_data.columns[-1]
        target_name = ['class', 'Class', 'target']

        if any(target_name) not in self.train_data.columns:
            label = last_name
        else:
            label = [x for x in self.train_data.columns if x in target_name][0]

        predictor = TabularPredictor(label=label).fit(train_data=self.train_data)

        return predictor

    def predict(self, predictor=None):
        start_time = timeit.default_timer()
        predictions = predictor.predict(self.test_data)
        inference = timeit.default_timer() - start_time

        target = self.test_data.iloc[:, -1]
        prediction_proba = predictor.predict_proba(self.test_data)

        return predictions, prediction_proba, target, inference

    @exception_decorator
    def perform_experiment(self):
        predictor = self.fit()
        predictions, prediction_proba, target, inference = self.predict(predictor)
        fit_report = predictor.fit_summary()

        return fit_report, predictions, prediction_proba, target, inference
