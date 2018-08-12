Interestingly `cumsum()` isn't doing what you expected, but `sum()` still gives the answer that I think you're concerned about:

```python
>>> pca_copy.explained_variance_ratio_
array([0.01209829, 0.00912749, 0.00900933, 0.00837896, 0.00720981,
       0.00694025, 0.00646099, 0.00549264, 0.00538648, 0.00519651,
       0.00490464, 0.00439539, 0.00424461, 0.00420902, 0.00412245,
       0.00397536])
>>> pca_copy.explained_variance_ratio_.cumsum()
array([0.01209829, 0.02122578, 0.03023511, 0.03861407, 0.04582388,
       0.05276413, 0.05922511, 0.06471775, 0.07010423, 0.07530074,
       0.08020538, 0.08460077, 0.08884538, 0.0930544 , 0.09717685,
       0.10115221])
>>> pca_copy.explained_variance_ratio_.cumsum()[-1]
0.10115221
>>> pca_copy.explained_variance_ratio_.sum()
0.10115221
```

That `0.1` total "explained variance" by all the 16 best features (components) created by PCA is still much lower than the `1.0` you were hoping for. But the reason is that you've summed the explained variance contributed by only 16 of the 9232 possible components (the number of words in your vocabulary). Some information is lost when you create linear combinations of the 9232 features to produce 16 components/features. Unless your 9232 features can be divided into 16 groups of features that are all perfectly correlated (exactly equal to each other, so that every time one word is used another word is used in the same document) then there's no way your sum would ever equal 1.0.

Before I realized this, I compared the explained variance to the imbalance in the training set and the accuracy achievable with a single-feature model. But none of this is really relevant to your question:

```python
>>> sms.spam.mean()
0.1319
>>> pca_copy.explained_variance_ratio_.sum() / sms.spam.mean()
0.7669
>>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
>>> lda = LDA()
>>> lda = lda.fit(tfidf_docs, sms.spam)
>>> lda.score(tfidf_docs, sms.spam)
1.0
>>> lda_pca = LDA()
>>> lda_pca = lda_pca.fit(pca16_topic_vectors, sms.spam)
>>> lda_pca.score(pca16_topic_vectors, sms.spam)
0.9568
>>> lda_pca1 = lda_pca.fit(pca16_topic_vectors[:,0].reshape((-1,1)), sms.spam)
>>> lda_pca1.score(pca16_topic_vectors[:,0].reshape((-1,1)), sms.spam)
0.8681  # this is the best you can do if you chose only the best component to make your predictions
```

86.8% correlation is the best you could do if you had only one PCA component as your feature and you scaled/thresholded that feature optimally to maximize the correlation with your target variable.
