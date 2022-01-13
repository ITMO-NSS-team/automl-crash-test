import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

from data import get_train_data, get_test_data


def run_lama():
    """ Launch LightAutoML (LAMA) framework for binary classification task """
    columns = ['f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_7', 'f_8', 'f_9', 'f_10', 'f_11']
    train_features, train_target = get_train_data()
    test_features, test_target = get_test_data()

    df_train = pd.DataFrame(train_features, columns=columns)
    df_train['target'] = train_target

    # Get rid of row with nan in target - because it causes error
    df_train = df_train.iloc[:-1]

    df_test = pd.DataFrame(test_features, columns=columns)
    df_test['target'] = test_target

    automl = TabularAutoML(task=Task(name='binary',
                                     metric=roc_auc_score))
    automl.fit_predict(df_train, roles={'target': 'target'})
    test_pred = automl.predict(df_test)

    roc_auc = roc_auc_score(test_target, test_pred.data, multi_class="ovr")
    print(f'ROC AUC score: {roc_auc:.3f}')


if __name__ == '__main__':
    run_lama()
