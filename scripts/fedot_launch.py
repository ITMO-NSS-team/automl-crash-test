from fedot.api.main import Fedot
from sklearn.metrics import classification_report, roc_auc_score

from data import get_train_data, get_test_data


def run_fedot(timeout: float = 1):
    """ Launch FEDOT AutoML framework for binary classification task

    :param timeout: time for optimization in minutes
    """
    train_features, train_target = get_train_data()
    test_features, test_target = get_test_data()

    # Task selection, initialisation of the framework
    fedot_model = Fedot(problem='classification', timeout=timeout)

    # Fit model
    obtained_pipeline = fedot_model.fit(features=train_features, target=train_target)
    obtained_pipeline.show()

    # Evaluate the prediction with test data
    predict = fedot_model.predict(test_features)
    predict_probs = fedot_model.predict_proba(test_features)

    print(classification_report(test_target, predict))
    roc_auc = roc_auc_score(test_target, predict_probs, multi_class="ovr")
    print(f'ROC AUC score: {roc_auc:.3f}')


if __name__ == '__main__':
    run_fedot(timeout=0.1)
