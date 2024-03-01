# NOTES #

notes and important links for alfatraining course in machine learning, in 2023  
(version: 00:00 01.03.2024) 


Machine Learning Book: Jörg Frochte, 2020, "Maschinelles Lernen — Grundlagen und Algorithmen in Python"  

* [Machine Learning Book web](https://joerg.frochte.de/books/) 
* [Machine Learning Book Erratum in 2020 as PDF](https://joerg.frochte.de/download/Erratum2020.pdf)  
* [Codes from the book in zip file](https://joerg.frochte.de/download/MLBuchCode.zip)  


Data preparation: Imputation, how to deal with missing data/values  

* [sklearn SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)  
* [sklearn OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)  
* [sklearn OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)  


Data preparation: scaling  

* [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html), for scaling based on minimum and maximum    
* [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), for scaling based on mean and standard deviation  


Data preparation: user-defined transformer  

* [FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)  
* [BaseEstimator](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html)  
* [TransformerMixin](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html)  
 
 
Checking:  

* the input is [checked to be a non-empty 2D array containing only finite values](https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_array.html)  
* [Checks if the estimator is fitted](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html) by verifying the presence of fitted attributes  


Building a machine learning model with sklearn (scikit-learn):  

* [Pipeline in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to sequentially apply a list of transforms and a final estimator  
* [LinearRegression in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) as an ordinary least squares Linear Regression  
* [DecisionTreeRegressor in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) as a decision tree regressor  
* [Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) using metrics and scoring for quantifying the quality of predictions  


Cross Validation, with sklearn (scikit-learn):  

* [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) for evaluating estimator performance  
* [cross_val_score in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) to evaluate a score by cross-validation  
* [cross_validate in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) to evaluate metric(s) by cross-validation and also record fit/score times  
* The [scoring parameter](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) for defining model evaluation rules  


Random Forest:  

* [RandomForestRegressor in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) as a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting  
* [GridSearchCV in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) as an exhaustive search over specified parameter values for an estimator  
* [Random Forest](https://en.wikipedia.org/wiki/Random_forest), on wikipedia  


Distribution:  

* [Galton board](https://en.wikipedia.org/wiki/Galton_board) to demonstrate the central limit theorem, especially the binomial distribution  
* [Galtonbrett](https://de.wikipedia.org/wiki/Galtonbrett) ist ein mechanisches Modell zur Demonstration und Veranschaulichung der Binomialverteilung  
* [Log-normal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution), on wikipedia  
* [Logarithmische Normalverteilung](https://de.wikipedia.org/wiki/Logarithmische_Normalverteilung), on german wikipedia  


Conditional Probability:  

* [Conditional probability](https://en.wikipedia.org/wiki/Conditional_probability), on wikipedia  
* [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem), on wikipedia  
* [Bayesian inference](https://en.wikipedia.org/wiki/Bayesian_inference), on wikipedia  
* [Bayesian probability](https://en.wikipedia.org/wiki/Bayesian_probability), on wikipedia  
* [Chain rule, in probability](https://en.wikipedia.org/wiki/Chain_rule_(probability)), on wikipedia  


Bayesian, with sklearn (scikit-learn):  

* [Naive Bayes algorithms](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes) as supervised learning methods based on applying Bayes’ theorem with strong (naive) feature independence assumptions  
* [Naive Bayes methods](https://scikit-learn.org/stable/modules/naive_bayes.html) are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable  
* [BernoulliNB in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB) as a Naive Bayes classifier for multivariate Bernoulli models  
* [CategoricalNB in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html#sklearn.naive_bayes.CategoricalNB) as a Naive Bayes classifier for categorical features  
* [GaussianNB in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) as a Gaussian Naive Bayes classifier  


Python Data Science Handbook, from Jake VanderPlas  

* [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/), on github page  
* [Linear Regression](https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html), from Python Data Science Handbook  


Newton's Method:  

* [Newtonverfahren](https://de.wikipedia.org/wiki/Newtonverfahren) oder Newton-Raphson-Verfahren, on german wikipedia  
* [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) or Newton–Raphson method, on wikipedia  
* [Metode Newton](https://id.wikipedia.org/wiki/Metode_Newton) atau metode Newton–Raphson, di wikipedia Indonesia  
* [Gauß-Newton-Verfahren](https://de.wikipedia.org/wiki/Gau%C3%9F-Newton-Verfahren), on german wikipedia  
* [Gauss–Newton algorithm](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm), on wikipedia  
* [Algoritma Gauss-Newton](https://id.wikipedia.org/wiki/Algoritma_Gauss-Newton), di wikipedia Indonesia  


Polynomials, with sklearn (scikit-learn):  

* [PolynomialFeatures in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) to generate polynomial and interaction features  


Numpy:  

* [numpy.newaxis](https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis) as convenient alias for None, useful for indexing arrays  
* [How do I use np.newaxis?](https://stackoverflow.com/questions/29241056/how-do-i-use-np-newaxis)  
* [numpy.dot](https://numpy.org/doc/stable/reference/generated/numpy.dot.html) as Dot product of two arrays  
* [numpy.arange](https://numpy.org/doc/stable/reference/generated/numpy.arange.html) to return evenly spaced values within a given interval  
* [numpy.linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html) to return evenly spaced numbers over a specified interval  
* [numpy.logspace](https://numpy.org/doc/stable/reference/generated/numpy.logspace.html) to return numbers spaced evenly on a log scale  


Train and test data split:  

* [train_test_split in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) to split arrays or matrices into random train and test subsets  


Linear Regression / Linear Model, with sklearn (scikit-learn):  

* [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)  
* list of [Linear Models in sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)  
* [LinearRegression in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), as Ordinary least squares Linear Regression  
* [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso), as Linear Model trained with L1 prior as regularizer (aka the Lasso)  
* [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge), as Linear least squares with l2 regularization  
* [LogisticRegression in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), as logit, MaxEnt classifier  


Logistic Regression:  

* [Logit and Expit](https://statisticaloddsandends.wordpress.com/2020/06/24/logit-and-expit/)  
* [Logit](https://en.wikipedia.org/wiki/Logit), on wikipedia  
* [Logit](https://de.wikipedia.org/wiki/Logit), on german wikipedia  
* [Probit](https://en.wikipedia.org/wiki/Probit), on wikipedia  
* [Probit](https://de.wikipedia.org/wiki/Probit), on german wikipedia  
* [Ridit scoring](https://en.wikipedia.org/wiki/Ridit_scoring), on wikipedia  


k-Nearest Neighbors (k-NN):  

* [Nächste-Nachbarn-Klassifikation](https://www.python-kurs.eu/naechste_nachbarn_klassifikation.php)  
* [Nächste-Nachbarn-Klassifikation mit sklearn](https://www.python-kurs.eu/naechste_nachbarn_klassifikation_sklearn.php)  
* [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm), on wikipedia  
* [Nächste-Nachbarn-Klassifikation](https://de.wikipedia.org/wiki/N%C3%A4chste-Nachbarn-Klassifikation), on german wikipedia  
* [Algoritma k tetangga terdekat](https://id.wikipedia.org/wiki/Algoritme_k_tetangga_terdekat), di wikipedia Indonesia  


k-Nearest Neighbors (k-NN), with sklearn (scikit-learn):  

* [Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)  
* list of [Nearest Neighbors algorithms in sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors)  
* [KNeighborsClassifier in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)  
* [BallTree in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html#sklearn.neighbors.BallTree)  
* [KDTree in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree)  


Distances:  

* [distance_metrics, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics)  


Mahalanobis Distances:  

* [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance), on wikipedia  
* [Mahalanobis-Abstand](https://de.wikipedia.org/wiki/Mahalanobis-Abstand), on german wikipedia  
* [Jarak Mahalanobis](https://id.wikipedia.org/wiki/Jarak_Mahalanobis), di wikipedia Indonesia  


Haversine Distances, on a sphere as longitude and latitude:  

* [haversine_distances, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html)  
* [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula), on wikipedia  


Cosine Distances:  

* [cosine_distances, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_distances.html#sklearn.metrics.pairwise.cosine_distances) is defined as 1.0 minus the cosine similarity  
* [cosine_similarity, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html#sklearn.metrics.pairwise.cosine_similarity) computes similarity as the normalized dot product of X and Y  
* [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity), on wikipedia  
* [Kosinus-Ähnlichkeit](https://de.wikipedia.org/wiki/Kosinus-%C3%84hnlichkeit), on german wikipedia  


Handling anomalies and missing data, in kNN with scikit-learn (sklearn):  

* [LocalOutlierFactor, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor), as an Unsupervised Outlier Detection using the Local Outlier Factor (LOF)  
* [RadiusNeighborsClassifier, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html), as a Classifier implementing a vote among neighbors within a given radius  
* [KNNImputer, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html), as Imputation for completing missing values using k-Nearest Neighbors  


Data sets in scikit-learn (sklearn):  

* list of [Datasets, in sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets)  
* [load_iris, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris), to load and return the iris dataset, as a classic and very easy multi-class classification dataset  
* [load_digits, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), to Load and return the UCI ML hand-written digits dataset (classification)  


Decision Tree: Gini, Entropy, and Information Gain  

* [Entropie (Informationstheorie)](https://de.wikipedia.org/wiki/Entropie_(Informationstheorie)), on german wikipedia  
* [Entropy (information theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory)), on wikipedia  
* [Entropi (teori informasi)](https://id.wikipedia.org/wiki/Entropi_(teori_informasi)), di wikipedia Indonesia  
* [Informationsgehalt](https://de.wikipedia.org/wiki/Informationsgehalt), on german wikipedia  
* [Information content](https://en.wikipedia.org/wiki/Information_content), on wikipedia  
* [Information gain (decision tree)](https://en.wikipedia.org/wiki/Information_gain_(decision_tree)), on wikipedia  
* [Entropie - Von Kühltürmen und der Unumkehrbarkeit der Dinge](https://www.youtube.com/watch?v=z64PJwXy--8), on youtube (13 minutes)  
* [Decision Trees: Gini vs Entropy](https://quantdare.com/decision-trees-gini-vs-entropy/)  
* [Entropy and Information Gain, in Machine Learning](https://www.section.io/engineering-education/entropy-information-gain-machine-learning/)  
* [How Decision tree classification and regression algorithm works](https://pro.arcgis.com/en/pro-app/latest/tool-reference/geoai/how-decision-tree-classification-and-regression-works.htm)  


[![Entropie - Von Kühltürmen und der Unumkehrbarkeit der Dinge](https://img.youtube.com/vi/z64PJwXy--8/0.jpg)](https://www.youtube.com/watch?v=z64PJwXy--8)


Decision Tree:  

* [Decision Trees, in sklearn](https://scikit-learn.org/stable/modules/tree.html)  
* [Decision tree learning](https://en.wikipedia.org/wiki/Decision_tree_learning), on wikipedia  
* [sklearn.tree: Decision Trees, in sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree)  
* [DecisionTreeClassifier, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)  
* [DecisionTreeRegressor, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)  


Otsu's method, to perform automatic image thresholding:  

* [Otsu's method](https://en.wikipedia.org/wiki/Otsu%27s_method), on wikipedia  
* [Metode Otsu](https://id.wikipedia.org/wiki/Metode_Otsu), di wikipedia Indonesia  


Visualization of tree:  

* [plot_tree, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html#sklearn.tree.plot_tree), to plot a decision tree  
* [export_graphviz, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html#sklearn.tree.export_graphviz), to export a decision tree in DOT format  
* [DOT (graph description language)](https://en.wikipedia.org/wiki/DOT_(graph_description_language)), on wikipedia  
* [DOT (Graphviz)](https://de.wikipedia.org/wiki/DOT_(Graphviz)), on german wikipedia  
* [DOT (bahasa yang mendeskripsikan grafik)](https://id.wikipedia.org/wiki/DOT_(bahasa_yang_mendeskripsikan_grafik)), di wikipedia Indonesia  


Support Vector Machines (SVM), from Jake VanderPlas "Python Data Science Handbook":  

* [In Depth: Support Vector Machines](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html), on github page  
* [In Depth: Support Vector Machines](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.07-Support-Vector-Machines.ipynb), as examples on github repository  


SVC = SVM classifier  

* [Support Vector Machine (SVM), in sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)  
* [SVC, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), for Support Vector Classification, based on libsvm  
* [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine), on wikipedia  
* [Support Vector Machine](https://de.wikipedia.org/wiki/Support_Vector_Machine), on german wikipedia  
* [Support Vector Machine](https://id.wikipedia.org/wiki/Support-vector_machine), di wikipedia Indonesia  
* [LIBSVM: A Library for Support Vector Machines](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf), in PDF  


Gaussian Process and Radial Basis Function kernel:  

* [RBF, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html), as Radial basis function kernel (aka squared-exponential kernel)  
* [Gaussian processes, in sklearn](https://scikit-learn.org/stable/modules/gaussian_process.html)  
* [the kernel trick](https://en.wikipedia.org/wiki/Kernel_method#Mathematics:_the_kernel_trick), on wikipedia  
* [Kernel-Methode](https://de.wikipedia.org/wiki/Kernel-Methode), on german wikipedia  
* [Radial basis function kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel), on wikipedia  


Twelfth day discussion links:  

* [make_blobs, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html), to generate isotropic Gaussian blobs for clustering  
* [NumPy for MATLAB users](https://numpy.org/devdocs/user/numpy-for-matlab-users.html), containing notations for matrix and vector multiplication  


Grid Search Cross Validation (GridSearchCV):  

* [GridSearchCV, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)  
* [RandomizedSearchCV, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)  
* [Tuning the hyper-parameters of an estimator, with Grid Search, in sklearn](https://scikit-learn.org/stable/modules/grid_search.html#grid-search)  
* [End-to-end Machine Learning project](https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb), on github about the Geron's book on Machine Learning   


Deep Learning Book: A. Geron, 2023, "Praxiseinstieg Machine Learning mit Scikit-Learn, Keras und TensorFlow"  

* [Machine Learning Book web](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)  
* [Machine Learning Book](https://homl.info/er3)  
* [Codes from the book in github](https://github.com/ageron/handson-ml3)  
* [other deep learning notes on github](https://github.com/iscab/belajar_python/blob/main/Course2023_alfatraining_Deep_Learning/my_notes/notes.md)  
* [other deep learning notes on bitbucket](https://bitbucket.org/iscab/alfatraining_2023_deep_learning/src/master/my_notes/notes.md)  


Model evaluation, in scikit-learn (sklearn):  

* [scoring parameter, in sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter), for defining model evaluation rules  
* [scoring, in sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring), for defining your scoring strategy from metric functions  
* [Classification metrics, in sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)  
* list of [Metrics, in sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)  
* [Metrics and scoring, in sklearn: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)  

```
D = pdist(XRauschen, metric='euclidean')
Z = linkage(D, myMethod, metric='euclidean')
c = fcluster(Z,2,criterion='maxclust')  
```


Neural Network, in scikit-learn (sklearn):  

* [Neural network models (supervised), in sklearn](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)  
* [MLPClassifier, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), as Multi-layer Perceptron classifier, by optimizing the log-loss function using LBFGS or stochastic gradient descent (SGD)  
* [MLPRegressor, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html), as Multi-layer Perceptron regressor, by optimizing the squared error using LBFGS or stochastic gradient descent (SGD)  


lbfgs = Limited-memory BGFS: Broyden-Fletcher-Goldfarb-Shanno algorithm  

* BGFS = Broyden-Fletcher-Goldfarb-Shanno algorithm  
* [Linear Models, in sklearn](https://scikit-learn.org/stable/modules/linear_model.html)  
* [Broyden–Fletcher–Goldfarb–Shanno algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm), on wikipedia  
* [Broyden–Fletcher–Goldfarb–Shanno (BFGS) Verfahren](https://de.wikipedia.org/wiki/BFGS-Verfahren), on german wikipedia  
* [Limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS), on wikipedia  


Random Forest, in scikit-learn (sklearn):  

* [VotingClassifier, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html), as Soft Voting/Majority Rule classifier for unfitted estimators  
* [RandomForestClassifier, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), as a random forest classifier  
* [RandomForestRegressor, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), as a random forest regressor  
* [Random forests and other randomized tree ensembles, in sklearn](https://scikit-learn.org/stable/modules/ensemble.html#forest)  
* [Using the VotingClassifier with GridSearchCV, in sklearn](https://scikit-learn.org/stable/modules/ensemble.html#using-the-votingclassifier-with-gridsearchcv), in order to tune the hyperparameters of the individual estimators  
* [Ensembles, in sklearn](https://scikit-learn.org/stable/modules/ensemble.html): Gradient boosting, random forests, bagging, voting, stacking  


Neural Network:  

* [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901), on arXiv  
* [Your Deep Learning Journey](https://github.com/fastai/fastbook/blob/master/01_intro.ipynb), from fastai book, on github  
* [The fastai book](https://github.com/fastai/fastbook), on github, using PyTorch  
* [Geometric Intuition for Training Neural Networks](https://www.youtube.com/watch?v=Z_MA8CWKxFU&t=0s), on youtube (31 minutes)   
* [Watching Neural Networks Learn](https://www.youtube.com/watch?v=TkwXa7Cvfr8&t=0s), on youtube (26 minutes)  


[![Geometric Intuition for Training Neural Networks](https://img.youtube.com/vi/Z_MA8CWKxFU/0.jpg)](https://www.youtube.com/watch?v=Z_MA8CWKxFU&t=0s)


[![Watching Neural Networks Learn](https://img.youtube.com/vi/TkwXa7Cvfr8/0.jpg)](https://www.youtube.com/watch?v=TkwXa7Cvfr8&t=0s)


compose, in scikit-learn (sklearn):  

* sklearn.compose as Meta-estimators for building composite models with transformers  
* [TransformedTargetRegressor, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html), as Meta-estimator to regress on a transformed target  
* list of [Composite Estimators, in sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.compose)  


Feature Selection, in scikit-learn (sklearn):  

* list of [Feature Selection algorithms, in sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection)  
* [feature_selection, in sklearn](https://scikit-learn.org/stable/modules/feature_selection.html), for feature selection/dimensionality reduction on sample sets, either to improve estimators’ accuracy scores or to boost their performance on very high-dimensional dataset  
* [VarianceThreshold, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html), as a Feature Selector that removes all low-variance features  
* [RFE, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html), for feature ranking with recursive feature elimination (RFE), to select features by recursively considering smaller and smaller sets of features, until the least important features are pruned from current set of features, and the desired number of features to select is eventually reached  
* [plot_rfe_digits, in sklearn](https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html), as a recursive feature elimination example showing the relevance of pixels in a digit classification task  
* [plot_rfe_with_cross_validation, in sklearn](https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html), as A Recursive Feature Elimination (RFE) example with automatic tuning of the number of features selected with cross-validation  


Principal component analysis (PCA), in scikit-learn (sklearn):  

* [PCA, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), as Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. The input data is centered but not scaled for each feature before applying the SVD  
* [KernelPCA, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html), as Non-linear dimensionality reduction through the use of kernels  
* list of [matrix decomposition algorithms, in sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition), including among others PCA, NMF or ICA  
* [Decomposing signals in components (matrix factorization problems), in sklearn](https://scikit-learn.org/stable/modules/decomposition.html#decompositions)  


Factor analysis and ICA, in scikit-learn (sklearn):  

* [FactorAnalysis, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html), as A simple linear generative model with Gaussian latent variables  
* [FastICA, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html), as a fast algorithm for Independent Component Analysis  


Eigen values, eigen vectors, and rotation:  

* [Eigenwerte und Eigenvektoren](https://www.mathebibel.de/eigenwerte-eigenvektoren), on Mathebibel  
* [Eigenwerte berechnen](https://www.mathebibel.de/eigenwerte-berechnen), on Mathebibel  
* [Rotationsverfahren (Statistik)](https://de.wikipedia.org/wiki/Rotationsverfahren_(Statistik)), on german wikipedia  
* [Varimax rotation](https://en.wikipedia.org/wiki/Varimax_rotation), on wikipedia  
* [Rotation methods, in factor analysis](https://en.wikipedia.org/wiki/Factor_analysis#Rotation_methods), on wikipedia  
* [Factor analysis](https://en.wikipedia.org/wiki/Factor_analysis), on wikipedia  
* [Faktorenanalyse](https://de.wikipedia.org/wiki/Faktorenanalyse), on german wikipedia  
* [Analisis faktor](https://id.wikipedia.org/wiki/Analisis_faktor), di wikipedia Indonesia  


Manifold, in scikit-learn (sklearn):  

* [TSNE, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), as T-distributed Stochastic Neighbor Embedding, a tool to visualize high-dimensional data  
* TSNE converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data  
* list of [Manifold Learning, in sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold), as implementation of data embedding techniques  
* [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), on wikipedia  
* [Kullback-Leibler-Divergenz](https://de.wikipedia.org/wiki/Kullback-Leibler-Divergenz), on german wikipedia  


Clustering, in scikit-learn (sklearn):  

* [Clustering, in sklearn](https://scikit-learn.org/stable/modules/clustering.html)  
* list of [unsupervised clustering algorithms, in sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster)  


K-Means Clustering:  

* [KMeans, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)  
* [K-means, in sklearn](https://scikit-learn.org/stable/modules/clustering.html#k-means)  
* [Visualizing K-Means Clustering](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)  
* [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering), on wikipedia  
* [k-Means-Algorithmus](https://de.wikipedia.org/wiki/K-Means-Algorithmus), on german wikipedia  
* [Pengklasteran k rata-rata](https://id.wikipedia.org/wiki/Pengklasteran_k_rata-rata), di wikipedia Indonesia  
* [Lapidot, 2008, "Convergence problems of Mahalanobis distance-based k-means clustering", IEEE ICSEE](https://ieeexplore.ieee.org/document/8646138)  


Density-based Spatial Clustering of Applications with Noise (DBSCAN):  

* [DBSCAN, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)  
* [Visualizing DBSCAN Clustering](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)  
* [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN), on wikipedia  
* [DBSCAN](https://de.wikipedia.org/wiki/DBSCAN), on german wikipedia  


Hierarchical clustering:  

* [AgglomerativeClustering, in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html), to recursively merge pair of clusters of sample data; by using linkage distance  
* [Plot Hierarchical Clustering Dendrogram, with sklearn and scipy](https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html)  
* [Hierarchical clustering: structured vs unstructured ward](https://scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html)  
* [Hierarchical clustering, in sklearn](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)  
* [Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering), on wikipedia  
* [Hierarchische Clusteranalyse](https://de.wikipedia.org/wiki/Hierarchische_Clusteranalyse), on german wikipedia  


Kneed:  

* [kneed, on pypi](https://pypi.org/project/kneed/), for Knee-point detection in Python 
* to implement the kneedle algorithm, to return the knee point of the function.  
* The knee point is the point of maximum curvature  
* [Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior, as paper in PDF](https://raghavan.usc.edu//papers/kneedle-simplex11.pdf)  
* [kneed documentation](https://kneed.readthedocs.io/en/stable/)  


Jupyter Notebook:  

* [How to Use Jupyter Notebook](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)  


### Project: Warum schmeckt Rotwein /Weißwein gut?  


Project details:  

* Voraussage von Bewertungen auf Grund von chemischen Eigenschaften.  
* Bitte nicht die Weinsorte (weiß,rot) voraussagen, das wäre zu leicht  
* Tipp: Wenn noch Zeit ist, mit verschiedenen Metriken experimentieren.  
* Die Schwierigkeit liegt in der Auswahl des Scores. Falls ein Algorithmus Metriken verwendet, müssen diese zum Sachgehalt der Daten passen.  


Wine Quality: Data Sets and Information  

* [Wine Quality: Data Sets](https://archive.ics.uci.edu/dataset/186/wine+quality), on UC Irvine website    
* [Wine Quality: Data Sets](https://www.kaggle.com/code/eisgandar/red-wine-quality-eda-classification/notebook), on Kaggle    
* [Red Wine Quality Analysis and Machine Learning Techniques using sklearn python libraries](https://www.youtube.com/watch?v=lHwESP3-Efg&t=0s), on youtube (30 minutes)  
* google drive links?  


[![Red Wine Quality Analysis and Machine Learning Techniques using sklearn python libraries](https://img.youtube.com/vi/lHwESP3-Efg/0.jpg)](https://www.youtube.com/watch?v=lHwESP3-Efg&t=0s)


check:  

* [alfatraining](https://www.alfatraining.de/gefoerderte-weiterbildung/) courses  
* [alfatraining](https://www.alfatraining.de/) website  
* detailed in [my private repository](https://bitbucket.org/iscab/alfatraining_2023_machine_learning/)  
* [this machine learning notes on github](https://github.com/iscab/belajar_python/blob/main/Course2023_alfatraining_Machine_Learning/my_notes/notes.md)  
* [this machine learning notes on bitbucket](https://bitbucket.org/iscab/alfatraining_2023_machine_learning/src/master/my_notes/notes.md)  
* [other deep learning notes on github](https://github.com/iscab/belajar_python/blob/main/Course2023_alfatraining_Deep_Learning/my_notes/notes.md)  
* [other deep learning notes on bitbucket](https://bitbucket.org/iscab/alfatraining_2023_deep_learning/src/master/my_notes/notes.md)  
* [other python notes on github](https://github.com/iscab/belajar_python/blob/main/Course2023_alfatraining_Python_Programmierung/my_notes/notes.md)  
* [other python notes on bitbucket](https://bitbucket.org/iscab/alfatraining_2023_python/src/master/my_notes/notes.md)  
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)  


version: 00:00 01.03.2024

# End of File

