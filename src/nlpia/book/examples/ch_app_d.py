import numpy as np


"""
----
>>> y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # <1>
>>> y_pred = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0])  # <2>
>>> true_positives = ((y_pred == y_true) & (y_pred == 1)).sum()
>>> true_positives  # <3>
4
>>> true_negatives = ((y_pred == y_true) & (y_pred == 0)).sum()
>>> true_negatives  # <4>
2
----
<1> `y_true` is a numpy array of the true (correct) class labels. Usually these are determined by a human
<2> `y_pred` is a numpy array of your model's predicted class labels (0 or 1)
<3> `true_positives` are the positive class labels (1) that your model got right (correctly labeled 1)
<4> `true_negatives` are the negative class labels (0) that your model got right (correctly labeled 0)
"""


y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # <1>
y_pred = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0])  # <2>
true_positives = ((y_pred == y_true) & (y_pred == 1)).sum()
true_positives  # <3>
# 4
true_negatives = ((y_pred == y_true) & (y_pred == 0)).sum()
true_negatives  # <4>
# 2

# <1> `y_true` is a numpy array of the true (correct) class labels. Usually these are determined by a human
# <2> `y_pred` is a numpy array of your model's predicted class labels (0 or 1)
# <3> `true_positives` are the positive class labels (1) that your model got right (correctly labeled 1)
# <4> `true_negatives` are the negative class labels (0) that your model got right (correctly labeled 0)


"""
----
>>> false_positives = ((y_pred != y_true) & (y_pred == 1)).sum()
>>> false_positives  # <1>
1
>>> false_negatives = ((y_pred != y_true) & (y_pred == 0)).sum()
>>> false_negatives  # <2>
3
----
<1> `false_positives` are the negative class examples (1) that were falsely labeled positive by your model (labeled 1 when they should be 0)
<2> `false_negatives` are the positive class examples (0) that were falsely labeled negative by your model (labeled 0 when they should be 1)

"""

false_positives = ((y_pred != y_true) & (y_pred == 1)).sum()
false_positives  # <1>
# 1
false_negatives = ((y_pred != y_true) & (y_pred == 0)).sum()
false_negatives  # <2>
# 3

# <1> `false_positives` are the negative class examples (1) that were falsely labeled positive by your model (labeled 1 when they should be 0)
# <2> `false_negatives` are the positive class examples (0) that were falsely labeled negative by your model (labeled 0 when they should be 1)
