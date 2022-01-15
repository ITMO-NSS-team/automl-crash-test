import numpy as np
import pandas as pd

COLUMNS = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
           'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10',
           'feature_11']


def get_train_data(as_pandas: bool = False):
    """ Generate table with features and one-dim array with target for binary classification task

    Column description by indices:
        0) int column with np.nan and np.inf values
        1) int column with nans more than 90%
        2) int-str column with categorical values (number of unique values = 12)
        3) int-str column the same as 2) column but with additional 13th label in the test part
        4) int column (number of unique values = 4) with nans
        5) str-int column with words and numerical cells
        6) str column with unique categories 'a', 'b', 'c' and spaces in labels.
        New category 'd' arise in the test part
        7) str binary column
        8) int binary column and nans
        9) str column with truly float values as strings
        10) str column with truly categorical values
        target column - contain nan
    """
    features = np.array([[0, np.nan, 1, 1, 1, 'monday', 'a ', 'true', 1, '0.1', 'a'],
                         [np.nan, 1, 2, 2, 0, 'tuesday', 'a', np.nan, 0, '1.3', np.inf],
                         [2, np.nan, 3, 3, np.nan, 3, 'b', 'false', 1, '?', 'c'],
                         [3, np.nan, 4, 4, 3.0, 4, '  a  ', 'true', 0, '1.3', '1'],
                         [4, np.nan, 5, 5.0, 0, 5, '   b ', np.nan, 0, '3.2', '2'],
                         [5, np.nan, 6, 6, 0, 6, '   c  ', 'false', 0, '8.0', '3'],
                         [6, np.inf, 7, 7, 0, 7, '    c  ', 'true', 1, '5.9', 'g'],
                         [7, np.inf, 8, 8, 1.0, 1, ' b   ', np.nan, 0, '4.9', 'h'],
                         [np.inf, np.inf, '9', '9', 2, 2, np.nan, 'true', 1, '2.0', 'i'],
                         [9, np.inf, '10', '10', 2, 3, ' a  ', 'false', 0, '1.3', 'j'],
                         [10, np.nan, 11.0, 11.0, 0, 4, 'a ', 'false', 0, '2.8', 'k'],
                         [11, np.nan, 12, 12, 2.0, 5, np.nan, 'false', 1, '3.2', 'l'],
                         [12, np.nan, 1, 1.0, 1.0, 6, ' c  ', 'false', 0, '11.1', 'm'],
                         [13, np.nan, 2, 2, 1, 7, ' c  ', 'true', np.nan, '12.5', 'n'],
                         [14, np.nan, 3, 3, 2.0, 1, 'b', 'false', np.nan, 'error', 'o'],
                         [15, np.nan, 4, 4, 1, 2, 'b  ', 'false', np.nan, '2.1', 'p'],
                         [16, np.nan, 5, 5, 0, 3, '   b       ', 'true', 1, '1.2', 'r'],
                         [17, 1, 6, 6, 0, 4, '  a      ', 'false', 0, '1.6', 's'],
                         [18, np.nan, 7, 7, 1, 5, ' b ', 'true', 1, '0.4', 'a'],
                         [19, np.nan, 8, 8, 1, 6, '  c ', 'false', 1, '5.1', 'c'],
                         [20, np.nan, 9, 9, 1, 'sunday', 'c ', 'false', 0, '5.5', 'a']],
                        dtype=object)

    target = np.array(['0', '0', '0', '0', '0', '1', '1', '0', 0, '0', 0, '0',
                       '1', 1, '0', '0', '0', '0', '0', '1', np.nan], dtype=object)

    if as_pandas:
        features = pd.DataFrame.from_records(features, columns=COLUMNS)
        features['target'] = target
    return features, target


def get_test_data(as_pandas: bool = False):
    """ Generate array with features and target for test """
    features = np.array([[21, 1, 5, 13, 0, 'tuesday', 'a ', 'true', 1, '4.2', 'r'],
                         [22, 1, 6, 14, 0, 3, 'a', 'true', 0, '4.1', 'a'],
                         [23, 1, 7, 9, 1, 4, ' b', 'true', 1, '0.2', 'a'],
                         [24, 1, 8, 8, 1, 5, 'a', 'false', 1, '0.5', 'c'],
                         [25, 1, 9, 7, 1, 6, 'd ', 'false', 0, '9.1', 'b']],
                        dtype=object)

    target = np.array(['0', '0', '0', '0', '1'], dtype=str)
    if as_pandas:
        features = pd.DataFrame.from_records(features, columns=COLUMNS)
        features['target'] = target
    return features, target
