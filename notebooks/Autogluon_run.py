from data.data import get_train_data, get_test_data
from scripts.autogluon_launch import AutoGluonRun

train_data, target_train = get_train_data(data_frame_flag=True)
test_data, target_test = get_test_data(data_frame_flag=True)
AG_runner = AutoGluonRun(data=(train_data, test_data))

if __name__ == '__main__':
    predictor = AG_runner.fit()
    predictions, prediction_proba, target, inference = AG_runner.predict(predictor)
    f = 2
