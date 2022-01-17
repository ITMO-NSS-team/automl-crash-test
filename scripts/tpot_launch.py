from sklearn.metrics import roc_auc_score
from tpot import TPOTClassifier
from data.data import get_train_data, get_test_data


def run_tpot(max_time_mins: float = 1):
    """ Launch TPOT framework for binary classification task """
    train_features, train_target = get_train_data()
    test_features, test_target = get_test_data()

    model = TPOTClassifier(max_time_mins=max_time_mins)
    model.fit(train_features, train_target)

    test_pred = model.predict(test_features)

    roc_auc = roc_auc_score(test_target, test_pred)
    print(f'ROC AUC score: {roc_auc:.3f}')


if __name__ == '__main__':
    run_tpot(max_time_mins=0.1)
