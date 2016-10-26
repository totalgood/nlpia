# Measuring Performance

The most important algorithm in a machine learning system is the model performance evaluation "algorithm."

If you don't know how good your prediction is likely to be, then you don't know whether you need to try to tune or change your machine learning algorithm.

## Resources

### Classification: RMSE, Gain/Lift, K-S Divergence, AUC, Gini, Concordance Ratio

This [medium article](http://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29)by  Tavish Srivastava  has a good explanation of some important performance metrics.

- RMSE: Root Mean Squared Error
  - works for both continuous values (regression) and categorical (classification)
- Confusion Matrix
  - works for both continuous values (regression) and categorical (classification)
  - several scalar "summary" metrics are possible
    - sensitivity
    - specificity
    - Mathews Correlation Coefficient (Phi)
      - equivalent to classical Pierson Correlation Co
- Gain and Lift
  - often used in A/B testing and marketing campaign planning
- Kolmogorov Smirnov
  - requires a continuous score and a classification label
    - can threshold a regression problem to create arbitrary classes to use with K-S metric
  - separation or difference between two distributions
- Area Under the ROC Curve
  - ROC: Radio Operating Characteristic (from information theory)
- Gini Coefficient
- Concordant/Discordant ratio

### Ranked Lists: RMSE, DCG, NDCG 

Ranked lists are tricky to evaluate. Usually you care about more than just the score or ranke RMSE. This Dato [blog post](http://blog.dato.com/how-to-evaluate-machine-learning-models-part-2b-ranking-and-regression-metrics?hsFormKey=3bfd17244b5ec353723ed4fc24134798&submissionGuid=732f66dd-90eb-4d6a-9bc4-97c656d29005#blog_subscription) by Alice Zheng is a really good overview of some common metrics used for assessing Machine Learning algorithms that produce ranked lists (like search results). It's also a good overview of regression and classification performance metrics.

[B White](https://gist.github.com/bwhite/3726239) coded up NDCG (as well as others in python) and lists these useful resouces for the "document retrieval" or "relevance" or "ranked list" prediction quality:

- [UT CS slides](http://www.cs.utexas.edu/~mooney/ir-course/slides/Evaluation.ppt)
- [NII tech report/paper](http://www.nii.ac.jp/TechReports/05-014E.pdf)
- [Stanford CS276 Handouts](http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf)
- [French Paper](http://hal.archives-ouvertes.fr/docs/00/72/67/60/PDF/07-busa-fekete.pdf)
- ["Learning to Rank for Information Retrieval"](http://research.microsoft.com/en-us/people/tyliu/letor-tutorial-sigir08.pdf) by Tie-Yan Liu

### NDCG: Normalized Discounted Cumulative Gain

The name's a mouthful, but it describes pretty well how to compute this metric. The [formulas on WikiPedia](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) translate into Python [like this](https://gist.github.com/bwhite/3726239)



## Detail

### RMSE

`np.sqrt(np.mean((predicted_value - true_value) ** 2))

This is the *1-sigma* error typically reported along with predictions. When people report a range of possible outcomes they typically multiple the 1-sigma value by 3 to get a *3-sigma* range that the outcome has a 99% chance of falling within.

This is what most people think of as error, such as when people are reporting values like [Gallup pole predictions](http://www.gallup.com/poll/158519/romney-obama-gallup-final-election-survey.aspx) of the presidential election last year: "Romney 49%, Obama 48% in Gallup's Final Election Survey... margin of error +/- 2%." [Obama won that election 51% to 47%](http://uselectionatlas.org/RESULTS/national.php) through [use of Machine Learning](https://www.technologyreview.com/s/509026/how-obamas-team-used-big-data-to-rally-voters/) to target the right states during the campaign. Dan Wagner, Obama's chief data scientist, put together a team of coders to help win the election. His algorithms and error metrics were better than Gallup's and the Romney campaign's.

Gallup's error was right at the edge of their predicted accuracy range, indicating that their error metric was probably inadequate. It didn't gage the true unpredictability of American voters in a close race well at all. But this isn't a fair criticism, because Gallup wasn't making a prediction of the election outcome or estimating the error in their prediction, but merely reporting the error range of the poll they took. They did radically adjust their polling procedures in 2012 as a result of this "outlier" result, though.

## Confusion Matrix

A [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) is just a truth table of matches and mismatches between a set of classifications. So it's often called:

-  contingency table (or matrix)
-  error matrix
-  agreement matrix ? (my own positive spin on "confusion")

(TODO: Example Graphic or Whiteboard Drawing here)

## [Project](../../huml/day2/)

### Confusion Matrix

See if you can build a confusion matrix.
Work your way from top to bottom with this feature set.

- Binary classification
  - matrix/dataframe with 4 values TP, TN, FP, FN
- Multiple Category Classification
  - NxN table of hits/misses in classification
- Aggregate statistics methods
- AUC (Area Under the Curve)

## Model Competence

It's important to continuously monitor your model's "competence" over time.

- the world may change
- your training data may have sample bias
  - seasonal
  - selection bias
  - nonrandom sampling

## Anomolies

Anomolous inputs indicate your model's competence may be drifting.

- measure feature vector distance from nearest other vector
- manhattan
- max difference along any dimension
- distance from centroids of your clusters
- make sure all clusters maintain a Gaussian distribution
