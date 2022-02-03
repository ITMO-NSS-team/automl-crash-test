import timeit
import os
from sklearn.metrics import roc_auc_score
from autogluon.tabular import TabularPredictor

from data.data import dict_with_different_versions_of_dataset


def run_autogluon():
    """ Launch AutoGluon framework for binary classification task """
    dataset_versions = dict_with_different_versions_of_dataset()

    for version_name, generators in dataset_versions.items():
        print(f'Version name {version_name}')
        try:
            get_train_data, get_test_data = generators
            df_train, _ = get_train_data(as_pandas=True)
            df_test, test_target = get_test_data(as_pandas=True)

            gluon_automl = TabularPredictor(label='target')
            gluon_automl.fit(train_data=df_train)
            predict_probs = gluon_automl.predict_proba(df_test)

            roc_auc = roc_auc_score(test_target, predict_probs)
            print(f'ROC AUC score: {roc_auc:.3f}')
        except Exception as ex:
            print(f'Launch failed due to {ex.__class__}, {ex.__str__()}')


if __name__ == '__main__':
    run_autogluon()
