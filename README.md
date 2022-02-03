# automl-crash-test
The repository contains material for testing AutoML frameworks and machine learning libraries on tabular data. There is a table with a very complex structure, so to run a library on it there is a need to make an effort

## Dataset description
Task: binary classification

Location: data/data.py 

Form: Two functions generating numpy arrays or pandas DataFrames for train and test

Useful Information:
* **column with index 1** - consists of a large number of gaps, if the column is used in training, it is likely that the algorithm will err heavily on the test;
* **column with index 5** - feature, it is the day of the week, which is sometimes marked with numbers and sometimes with a string. The pattern is that if it's a working day, it's always class 0; if it's a weekend, it's always class 1 in the target;
* **column with index 6** - Categories "a" and "b" always mean that the target will be class 0, any other values mean that it will be 1;
* **column with index 9** - The values are actually of the float type. If the values are less than 5.0, then target will be 0, if greater than 5.0, then target will be 1.

If the information from columns 5, 6 and 9 (at least from one column) is processed correctly, then the ROC AUC test metric will be 1.0. 

Good luck!

## Current frameworks statuses

The table shows the value of the ROC AUC metric for the test part

| Framework   | Version | Full dataset                                                                                                                                                                                                  | Without nan  <br> in target                                                                                                                                                                              | Without int  <br> in target                                                                                                                                                                              |
| :---------: | :-----: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| FEDOT       | 0\.5.1  | 1\.0                                                                                                                                                                                                          | 1\.0                                                                                                                                                                                                     | 1\.0                                                                                                                                                                                                     |
| LightAutoML | 0\.3.2  | AssertionError: <br> Nan in target detected                                                                                                                                                                   | TypeError: <br> '<' not supported between  <br> instances of 'int' and 'str'                                                                                                                             | 0\.5                                                                                                                                                                                                     |
| TPOT        | 0\.11.7 | TypeError: <br> ufunc 'isnan' not  <br> supported for the input types,  <br> and the inputs could not be  <br> safely coerced to any <br>  supported types  <br> according to the casting  <br> rule ''safe'' | TypeError: <br> ufunc 'isnan' not <br> supported for the input types, <br> and the inputs could not be <br> safely coerced to any  <br> supported types <br> according to the casting <br> rule ''safe'' | TypeError: <br> ufunc 'isnan' not <br> supported for the input types, <br> and the inputs could not be <br> safely coerced to any  <br> supported types <br> according to the casting <br> rule ''safe'' |
| Autogluon   | 0\.3.1  | -                                                                                                                                                                                                             | -                                                                                                                                                                                                        | -                                                                                                                                                                                                        |