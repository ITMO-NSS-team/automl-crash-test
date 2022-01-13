import numpy as np


def get_train_data():
    """ Generate table with features and one-dim array with target for binary classification task

    Column description by indices:
        0) int column with single np.nan value
        1) int column with nans more than 90%
        2) int-str column with categorical values (number of unique values = 12)
        3) int-str column the same as 2) column but with additional 13th label in the test part
        4) str-int column with words and numerical cells
        5) int column (number of unique values = 4)
        6) str column with unique categories 'a', 'b', 'c' and spaces in labels.
        New category 'd' arise in the test part
        7) str binary column
        8) int binary column and nans
        9) str column with truly float values as strings
        10) str column with truly categorical values
        target column - contain nan
    """
    features = np.array([[0, np.nan, 1, 1, 1, 'monday', 'a ', 'true', 1, '0.1', 'a'],
                         [np.nan, 5, 2, 2, 0, 'tuesday', 'b', np.nan, 0, '1.3', np.inf],
                         [2, np.nan, 3, 3, np.nan, 3, 'c', 'false', 1, '?', 'c'],
                         [3, np.nan, 4, 4, 3.0, 4, '  a  ', 'true', 0, '2.3', '1'],
                         [4, np.nan, 5, 5.0, 0, 5, '   b ', np.nan, 0, '3.2', '2'],
                         [5, np.nan, 6, 6, 0, 6, '   c  ', 'false', 0, '4.0', '3'],
                         [6, np.inf, 7, 7, 0, 7, '    a  ', 'true', 1, '5.9', 'g'],
                         [7, np.inf, 8, 8, 1.0, 1, ' b   ', np.nan, 0, '6.9', 'h'],
                         [np.inf, np.inf, '9', '9', 2, 2, np.nan, 'true', 1, '7.0', 'i'],
                         [9, np.inf, '10', '10', 2, 3, ' c  ', 'false', 0, '8.3', 'j'],
                         [10, np.nan, 11.0, 11.0, 0, 4, 'c ', 'false', 0, '9.8', 'k'],
                         [11, np.nan, 12, 12, 2.0, 5, np.nan, 'false', 1, '8.2', 'l'],
                         [12, np.nan, 1, 1.0, 1.0, 6, ' b  ', 'false', 0, '11.1', 'm'],
                         [13, np.nan, 2, 2, 1, 7, ' c  ', 'true', np.nan, '12.5', 'n'],
                         [14, np.nan, 3, 3, 2.0, 1, 'a', 'false', np.nan, 'error', 'o'],
                         [15, np.nan, 4, 4, 1, 2, 'a  ', 'false', np.nan, '13.1', 'p'],
                         [16, np.nan, 5, 12, 0, 3, '   a       ', 'true', 1, '16.2', 'r'],
                         [17, 3, 6, 10, 0, 4, '  b      ', 'false', 0, '1.6', 's'],
                         [18, np.nan, 7, 9, 1, 5, ' b ', 'true', 1, '0.4', 'a'],
                         [19, np.nan, 8, 8, 1, 6, '  b ', 'false', 1, '0.1', 'c'],
                         [20, np.nan, 9, 7, 1, 'saturday', 'a ', 'false', 0, '0.5', 'a']],
                        dtype=object)

    target = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, np.nan])

    return features, target


def get_test_data():
    """ Generate array with features and target for test """
    features = np.array([[21, 2, 5, 13, 0, 3, 'd', 'true', 1, '4.2', 'r'],
                         [22, 3, 6, 14, 0, 4, 'a', 'true', 0, '5.1', 'a'],
                         [23, 4, 7, 9, 1, 5, 'b', 'true', 1, '0.2', 'a'],
                         [24, 5, 8, 8, 1, 6, 'c', 'false', 1, '0.5', 'c'],
                         [25, 6, 9, 7, 1, 'saturday', 'a ', 'false', 0, '0.1', 'b']],
                        dtype=object)

    target = np.array([1, 0, 0, 0, 1])

    return features, target
