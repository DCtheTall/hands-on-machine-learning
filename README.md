# Hands On Machine Learning
## with Scikit-Learn and TensorFlow

<img width="200" src="./handson-cover.jpeg">

This repository is my notes and exercise solutions while I was reading
_Hands On Machine Learning with Scikit-Learn and TensorFlow_ by Aurélian Géron.

All of the code and notes are written in [Jupyter](https://jupyter.org/)
notebooks that are written to run in
[Google Colab](https://colab.research.google.com). This allows me to not
need to set up a virtual environment and install machine learning libraries
each time. Colab has most of the modules this book uses installed by default.
There is some Colab specific code in here that will not work on a
run-of-the-mill Jupyter Notebook kernel. There are aslo differences in how
mathematical formuals are rendered in Colab than in Github's `ipynb` renderer.

## Table of Contents

1. [Topics Covered](#topics-covered)
2. [Academic Papers Cited](#academic-papers-cited)
3. [License](#license)

## Topics Covered

### Chapter 2: End to End Machine Learning Project

#### `Housing.ipynb`
- Downloading data for a machine learning project.
- Inspecting data with `pandas`.
- Plotting histograms of the data.
- Splitting data into training and test sets.
- Discretizing continuous features.
- Visualizing geographic data.
- Computing the correlation between features with `pandas`.
- Visualizing relationships between features using scatter plots.
- Combining features to make new ones.
- Cleaning the data for machine learning.
- Handling categorical features using `OneHotEncoder`.
- Defining custom feature transformations.
- Scikit-Learn's `Pipeline` object.
- Computing the root mean squared error (RMSE) of a regression model.
- Cross validation.
- Grid searching hyperparameters with Scikit-Learn's `GridSearchCV`.
- Evaluating models with a test set.
- Training a Support Vector Machine (SVM) on the Housing dataset.
- Fine tuning a `RandomForestRegressor` using `RandomizedSearchCV`.
- Creating a pipeline for a machine learning model with the Housing dataset.

### Chapter 3: Classification

#### `MNIST.ipynb`
- The MNIST dataset.
- Training a binary classifier.
- Measuring accuracy of a classifier model.
- Confusion matrix.
- Precision and recall.
- F1 score.
- Precision/recall tradeoff.
- The receiver operating characteristic (ROC) curve.
- Multiclass classifier.
- Error analysis using the confusion matrix and plotting examples where the
model was wrong.
- Multilabel classification.
- Multioutput classification.

#### `SpamClassifier.ipynb`
- Downloading the email dataset.
- Processing email data in Python.
- Counting the most common words and symbols in text data.
- Viewing the headers in email data with Python.
- Parsing HTML into plaintext with Python.
- Transforming words into their stems with Scikit-Learn's `PorterStemmer`
class.
- Extracting URLs from text with Python.
- Defining a transformer class using Scikit-Learn which extracts the word
counts from email data.
- Defining a transformer class which transforms word counts into a vector
which can be used as an input to a machine learning model.
- Using `LogisticRegression` to classify emails as spam or ham.
- Evaluating the model's performance with cross-validation.

#### `Titanic.ipynb`

- Downloading the Titanic dataset from [Kaggle](https://www.kaggle.com/).
- Defining a `Pipeline` to transform the data from Kaggle into input for a
machine learning model.
- Train an `SGDClassifier` to determine if a passenger survived or died on
the Titanic.
- Evaluate the `SGDClassifier` with cross-validation.
- Do a grid search with `GridSearchCV` to fine tune the hyperparameters of
a `RandomForestClassifier`.

### Chapter 4: Training Models

#### `TrainingModels.ipynb`
- Linear Regression.
- Mean squared error (MSE).
- The normal equation and its computational complexity.
- Batch Gradient Descent.
- Learning rate.
- Stochastic Gradient Descent.
- Mini-Batch Gradient Descent.
- Polynomial regression with Scikit-Learn.
- Learning curves.
- Regularization.
- Ridge Regression with Scikit-Learn.
- Lasso Regression with Scikit-Learn.
- Elastic Net with Scikit-Learn.
- Early stopping.
- Logistic Regression.
- Decision boundaries.
- Softmax Regression with Scikit-Learn on the Iris dataset.
- Implementing Batch Gradient Descent and Softmax Regression without using
Scikit-Learn.

### Chapter 5: Support Vector Machines
- Linear Support Vector Machine (SVM) classification.
- Hard margin classification.
- Soft margin classification.
- Scikit-Learn's `LinearSVM` classifier.
- Nonlinear SVMs classification.
- Using polynomial kernels for SVM classification.
- Adding features using a similarity function and landmark instances.
- Gaussian Radial Bias Function (RBF).
- Using Gaussian RBF kernels for SVM classification.
- Computational complexity of SVMs.
- SVM Regression.
- Scikit-Learn's `LinearSVR` class.
- Nonlinear SVM regression using Scikit-Learn's `SVR` class.
- The SVM decision function.
- The training objective for an SVM for hard and soft margin classification.
- Quadratic programming.
- Solving the dual problem of a quadratic programming problem.

## Academic Papers Cited

## License