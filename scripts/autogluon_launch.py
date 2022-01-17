import timeit
import os
from sklearn.metrics import roc_auc_score
from autogluon.tabular import TabularPredictor

from data.data import get_train_data, get_test_data


def run_autogluon():
    """ Launch AutoGluon framework for binary classification task """
    df_train, _ = get_train_data(as_pandas=True)
    df_test, test_target = get_test_data(as_pandas=True)

    gluon_automl = TabularPredictor(label='target')
    gluon_automl.fit(train_data=df_train)
    predict = gluon_automl.predict_proba(df_test)

    roc_auc = roc_auc_score(test_target, predict_probs)
    print(f'ROC AUC score: {roc_auc:.3f}')


if __name__ == '__main__':
    run_autogluon()
