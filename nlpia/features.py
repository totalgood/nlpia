import pandas as pd
import numpy as np


def optimize_feature_power(df, output_column_name=None, exponents=[2., 1., .8, .5, .25, .1, .01]):
    """ Plot the correlation coefficient for various exponential scalings of input features

    >>> np.random.seed(314159)
    >>> df = pd.DataFrame()
    >>> df['output'] = np.random.randn(1000)
    >>> df['x10'] = df.output * 10
    >>> df['sq'] = df.output ** 2
    >>> df['sqrt'] = df.output ** .5
    >>> optimize_feature_power(df, output_column_name='output').round(2)
            x10    sq  sqrt
    power
    2.00  -0.08  1.00  0.83
    1.00   1.00 -0.08  0.97
    0.80   1.00  0.90  0.99
    0.50   0.97  0.83  1.00
    0.25   0.93  0.76  0.99
    0.10   0.89  0.71  0.97
    0.01   0.86  0.67  0.95

    Returns:
      DataFrame:
         columns are the input_columns from the source dataframe (df)
         rows are correlation with output for each attempted exponent used to scale the input features
    """
    output_column_name = list(df.columns)[-1] if output_column_name is None else output_column_name
    input_column_names = [colname for colname in df.columns if output_column_name != colname]
    results = np.zeros((len(exponents), len(input_column_names)))
    for rownum, exponent in enumerate(exponents):
        for colnum, column_name in enumerate(input_column_names):
            results[rownum, colnum] = (df[output_column_name] ** exponent).corr(df[column_name])
    results = pd.DataFrame(results, columns=input_column_names, index=pd.Series(exponents, name='power'))
    # results.plot(logx=True)
    return results
