from sklearn.metrics import roc_auc_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

from data.data import dict_with_different_versions_of_dataset


def run_lama():
    """ Launch LightAutoML (LAMA) framework for binary classification task """
    dataset_versions = dict_with_different_versions_of_dataset()

    for version_name, generators in dataset_versions.items():
        print(f'Version name {version_name}')
        try:
            get_train_data, get_test_data = generators
            df_train, _ = get_train_data(as_pandas=True)
            df_test, test_target = get_test_data(as_pandas=True)

            automl = TabularAutoML(task=Task(name='binary',
                                             metric=roc_auc_score))
            automl.fit_predict(df_train, roles={'target': 'target'})
            test_pred = automl.predict(df_test)

            roc_auc = roc_auc_score(test_target, test_pred.data)
            print(f'ROC AUC score: {roc_auc:.3f}')
        except Exception as ex:
            print(f'Launch failed due to {ex.__class__}, {ex.__str__()}')


if __name__ == '__main__':
    run_lama()
