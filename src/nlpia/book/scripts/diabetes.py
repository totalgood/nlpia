""" A brief introduction to machine learning and linear regression 

Almost a verbatum copy of the code from the
[sklearn tutorial](http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html).
Only difference is that Pandas DataFrames are used instead of the custom dataset format that sklearn uses.
This makes the example code reusable for other problems/datasets that the student might see online (XLS, CSV, and HTML tables).

The "diabetes" dataset is briefly described in the
[sklearn documentation](http://scikit-learn.org/stable/datasets/index.html#diabetes-dataset).
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# # Load the diabetes dataset and conert it to a dataframe
# from sklearn import datasets
# diabetes = datasets.load_diabetes()
# df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
# df['diabetes_severity'] = diabetes.target
# df.to_csv('diabetes.csv', index=None)

df = pd.read_csv('diabetes.csv')
print(df.describe())
#                 age           sex           bmi        ...                    s5            s6  diabetes_severity
# count  4.420000e+02  4.420000e+02  4.420000e+02        ...          4.420000e+02  4.420000e+02         442.000000
# mean  -3.634599e-16  1.296411e-16 -8.042209e-16        ...         -3.830854e-16 -3.411950e-16         152.133484
# std    4.761905e-02  4.761905e-02  4.761905e-02        ...          4.761905e-02  4.761905e-02          77.093005
# min   -1.072256e-01 -4.464164e-02 -9.027530e-02        ...         -1.260974e-01 -1.377672e-01          25.000000
# 25%   -3.729927e-02 -4.464164e-02 -3.422907e-02        ...         -3.324879e-02 -3.317903e-02          87.000000
# 50%    5.383060e-03 -4.464164e-02 -7.283766e-03        ...         -1.947634e-03 -1.077698e-03         140.500000
# 75%    3.807591e-02  5.068012e-02  3.124802e-02        ...          3.243323e-02  2.791705e-02         211.500000
# max    1.107267e-01  5.068012e-02  1.705552e-01        ...          1.335990e-01  1.356118e-01         346.000000


# Use only one feature, BMI (Body Mass Index)
diabetes_X = df[['bmi']]
diabetes_Y = df[['diabetes_severity']]

# Ignore the last 20 examples/samples in the trianing set those as the test examples
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes_Y[:-20]
diabetes_y_test = diabetes_Y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show(block=False)

