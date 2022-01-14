import timeit
from scripts.utils import ExceptionDecorator
from autogluon.tabular import TabularPredictor
from scripts.tabular_laucher import TabularLauncher


class AutoGluonRun(TabularLauncher):

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

    @ExceptionDecorator
    def perform_experiment(self):
        predictor = self.fit()
        predictions, prediction_proba, target, inference = self.predict(predictor)
        fit_report = predictor.fit_summary()

        return fit_report, predictions, prediction_proba, target, inference
