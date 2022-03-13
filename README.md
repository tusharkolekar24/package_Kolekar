# package-Kolekar 0.2

package-Kolekar is a Python library for dealing with regression problem statements. It provides gradient descent and Ensemble Learning base regression models to build efficient Machine Learning Models/Projects.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install package-Kolekar
```
## Main Features
Gradient Descent:
* Linear Regression
* Ridge Regression
* Lasso
* ElasticNet 

Ensemble Learning:
* Avg_Ensemble

## LinearRegressionModel

```python
from package_Kolekar.Gradient_Descent.LinearRegression import LinearRegressionModel
import pandas as pd

dy = pd.read_csv('')

# Define Model from package_Kolekar
model = LinearRegressionModel(learning_rate=0.3, max_iter=1000)

# Make the shower that the target feature should be attached at the end of the DataFrame.
# dy is a DataFrame that carries multiple independent features and a single target feature
X,y = model.data_preparation(dy)
print("Shape of X:",X.shape)
print("Shape of y:",y.shape)

# Normalized data using MinMaxScaler with default scaling range (0,1)
X_scaled , y_scaled = model.data_normalization(dy)

#spilting dataset into test train split with train size of 80% and test size of the 20%
X_train, X_test, y_train, y_test = model.data_split_train_test(X_scaled,y_scaled)

#Prepare Dataset and fit it to LinearRegressionModel
model.get_fit(X_train,y_train)

#Now perform Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#Evaluate the performance of the models.
result_train = model.performance_evaluation(y_train,y_train_pred)
print("MAE for train set       :",result_train[0])
print("MSE for train set       :",result_train[1])
print("RMSE for train set      :",result_train[2])
print("MAPE for train set      :",result_train[3])
print("R^2 score for train set :",result_train[4])

result_test = model.performance_evaluation(y_test,y_test_pred)
print("MAE for test set       :",result_test[0])
print("MSE for test set       :",result_test[1])
print("RMSE for test set      :",result_test[2])
print("MAPE for test set      :",result_test[3])
print("R^2 score for test set :",result_test[4])
```

## RidgeRegressionModel

```python
from package_Kolekar.Gradient_Descent.RidgeRegression import RidgeRegressionModel
import pandas as pd

dy = pd.read_csv('')

# Define Model from package_Kolekar
model = RidgeRegressionModel(learning_rate=0.3, max_iter=1000,l2_penalty=1)

# Data Preparation
X,y = model.data_preparation(dy)
print("Shape of X:",X.shape)
print("Shape of y:",y.shape)

# Normalized data using MinMaxScaler with default scaling range (0,1)
X_scaled , y_scaled = model.data_normalization(dy)

#spilting dataset into test train split with train size of 80% and test size of the 20%
X_train, X_test, y_train, y_test = model.data_split_train_test(X_scaled,y_scaled)

#Prepare Dataset and fit it to RidgeRegressionModel
model.get_fit(X_train,y_train)

#Now perform Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#Evaluate the performance of the models.
result_train = model.performance_evaluation(y_train,y_train_pred)
print("MAE for train set       :",result_train[0])
print("MSE for train set       :",result_train[1])
print("RMSE for train set      :",result_train[2])
print("MAPE for train set      :",result_train[3])
print("R^2 score for train set :",result_train[4])

result_test = model.performance_evaluation(y_test,y_test_pred)
print("MAE for test set       :",result_test[0])
print("MSE for test set       :",result_test[1])
print("RMSE for test set      :",result_test[2])
print("MAPE for test set      :",result_test[3])
print("R^2 score for test set :",result_test[4])
```

## LassoRegressionModel

```python
from package_Kolekar.Gradient_Descent.LassoRegression import LassoRegressionModel
import pandas as pd

dy = pd.read_csv('')

# Define Model from package_Kolekar
model = LassoRegressionModel(learning_rate=0.3, max_iter=1000,l1_penalty=0.1)

# Data Preparation
X,y = model.data_preparation(dy)
print("Shape of X:",X.shape)
print("Shape of y:",y.shape)

# Normalized data using MinMaxScaler with default scaling range (0,1)
X_scaled , y_scaled = model.data_normalization(dy)

#spilting dataset into test train split with train size of 80% and test size of the 20%
X_train, X_test, y_train, y_test = model.data_split_train_test(X_scaled,y_scaled)

#Prepare Dataset and fit it to LassoRegressionModel
model.get_fit(X_train,y_train)

#Now perform Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#Evaluate the performance of the models.
result_train = model.performance_evaluation(y_train,y_train_pred)
print("MAE for train set       :",result_train[0])
print("MSE for train set       :",result_train[1])
print("RMSE for train set      :",result_train[2])
print("MAPE for train set      :",result_train[3])
print("R^2 score for train set :",result_train[4])

result_test = model.performance_evaluation(y_test,y_test_pred)
print("MAE for test set       :",result_test[0])
print("MSE for test set       :",result_test[1])
print("RMSE for test set      :",result_test[2])
print("MAPE for test set      :",result_test[3])
print("R^2 score for test set :",result_test[4])
```
## ElasticNetRegressionModel

```python
from package_Kolekar.Gradient_Descent.ElasticNetRegression import ElasticNetRegressionModel
import pandas as pd

dy = pd.read_csv('')

# Define Model from package_Kolekar
model = ElasticNetRegressionModel(learning_rate=0.3, max_iter=1000,l1_penalty=0.1,l2_penalty=1e-10)

# Data Preparation
X,y = model.data_preparation(dy)
print("Shape of X:",X.shape)
print("Shape of y:",y.shape)

# Normalized data using MinMaxScaler with default scaling range (0,1)
X_scaled , y_scaled = model.data_normalization(dy)

#spilting dataset into test train split with train size of 80% and test size of the 20%
X_train, X_test, y_train, y_test = model.data_split_train_test(X_scaled,y_scaled)

#Prepare Dataset and fit it to ElasticNetRegressionModel
model.get_fit(X_train,y_train)

#Now perform Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#Evaluate the performance of the models.
result_train = model.performance_evaluation(y_train,y_train_pred)
print("MAE for train set       :",result_train[0])
print("MSE for train set       :",result_train[1])
print("RMSE for train set      :",result_train[2])
print("MAPE for train set      :",result_train[3])
print("R^2 score for train set :",result_train[4])

result_test = model.performance_evaluation(y_test,y_test_pred)
print("MAE for test set       :",result_test[0])
print("MSE for test set       :",result_test[1])
print("RMSE for test set      :",result_test[2])
print("MAPE for test set      :",result_test[3])
print("R^2 score for test set :",result_test[4])
```
## Average Ensemble
The average Ensemble technique is used that create multiple models and then combine them to produce improved results. For individual models, weights are assigned manually. 

```python
import pandas as pd
from package_Kolekar.Ensemble_Learning.Avg_Ensemble import Average_weight_Ensemble

dy = pd.read_csv('')

# Define Model from package_Kolekar
model = Average_weight_Ensemble()

# Data Preparation
X,y = model.data_preparation(dy)
print("Shape of X:",X.shape)
print("Shape of y:",y.shape)

# Normalized data using MinMaxScaler with default scaling range (0,1)
X_scaled , y_scaled = model.data_normalization(dy)

#spilting dataset into test train split with train size of 80% and test size of the 20%
X_train, X_test, y_train, y_test = model.data_split_train_test(X_scaled,y_scaled)

#Prepare Dataset and fit it to Average_weight_Ensemble
base_models = model.set_base_models()
print(base_models)

weight=[0.3,0.2,0.1,0.05, 0.05,0.15,0.1,0.05]

#Now perform Predictions
y_train_pred = model.get_weighted_Avg_technique(base_model=base_models, 
                              train_X=X_train, 
                              train_y=y_train, 
                              test_X = X_train,
                              weights=weight)

y_test_pred = model.get_weighted_Avg_technique(base_model=base_models, 
                              train_X=X_train, 
                              train_y=y_train, 
                              test_X = X_test,weights=weight)

#Evaluate the performance of the models.
result_train = model.performance_evaluation(y_train,y_train_pred)
print("MAE for train set       :",result_train[0])
print("MSE for train set       :",result_train[1])
print("RMSE for train set      :",result_train[2])
print("MAPE for train set      :",result_train[3])
print("R^2 score for train set :",result_train[4])

result_test = model.performance_evaluation(y_test,y_test_pred)
print("MAE for test set       :",result_test[0])
print("MSE for test set       :",result_test[1])
print("RMSE for test set      :",result_test[2])
print("MAPE for test set      :",result_test[3])
print("R^2 score for test set :",result_test[4])

```
## Weighted Average Ensemble
In this case, we 1st calculated weights and sequence of base_model before being assigned to Ensemble learning Model.

```python
import pandas as pd
from package_Kolekar.Ensemble_Learning.Avg_Ensemble import Average_weight_Ensemble

dy = pd.read_csv('')

# Define Model from package_Kolekar
model = Average_weight_Ensemble()

# Data Preparation
X,y = model.data_preparation(dy)
print("Shape of X:",X.shape)
print("Shape of y:",y.shape)

# Normalized data using MinMaxScaler with default scaling range (0,1)
X_scaled , y_scaled = model.data_normalization(dy)

#spilting dataset into test train split with train size of 80% and test size of the 20%
X_train, X_test, y_train, y_test = model.data_split_train_test(X_scaled,y_scaled)

#Prepare Dataset and fit it to Average_weight_Ensemble
base_models = model.set_base_models()
print(base_models)

# calculated weights and sequence of the base model.
summary = model.get_weights(threshold=0.85,
                  base_model=base_models,
                  train_X=X_train, 
                  test_X=X_test,
                  train_y=y_train,
                  test_y=y_test)
print(summary)

weight=summary.weights.values
# Note: As per summary re-arange base_models sequence then only send further.

#Now perform Predictions
y_train_pred = model.get_weighted_Avg_technique(base_model=base_models, 
                              train_X=X_train, 
                              train_y=y_train, 
                              test_X = X_train,
                              weights=weight)

y_test_pred = model.get_weighted_Avg_technique(base_model=base_models, 
                              train_X=X_train, 
                              train_y=y_train, 
                              test_X = X_test,weights=weight)

#Evaluate the performance of the models.
result_train = model.performance_evaluation(y_train,y_train_pred)
print("MAE for train set       :",result_train[0])
print("MSE for train set       :",result_train[1])
print("RMSE for train set      :",result_train[2])
print("MAPE for train set      :",result_train[3])
print("R^2 score for train set :",result_train[4])

result_test = model.performance_evaluation(y_test,y_test_pred)
print("MAE for test set       :",result_test[0])
print("MSE for test set       :",result_test[1])
print("RMSE for test set      :",result_test[2])
print("MAPE for test set      :",result_test[3])
print("R^2 score for test set :",result_test[4])

```
## Contributing
Contributions in any form are welcome, including:
- Documentation improvements
- Additional tests
- New features to existing models
- New models

## License
[MIT](https://github.com/tusharkolekar24/package_Kolekar/blob/main/LICENSE)
