# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Decision tree classifier: Car safety

# Code and data taken from: https://www.kaggle.com/prashant111/decision-tree-classifier-tutorial

# ## Import libraries

# +
import boto3 # required in case we store the artifacts on s3
import category_encoders as ce
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import mlflow
import os
import warnings

from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# -

warnings.filterwarnings('ignore')
# %matplotlib inline

# ## Get MLFlow server URI

registry_uri = os.getenv('REGISTRY_URI')
if not registry_uri:
    raise Exception('REGISTRY_URI env variable should be defined on the system in order to log the generated model')

# ## Import dataset

# load dataset
data = 'car_evaluation.csv'
df = pd.read_csv(data, header=None)

# view dimensions of dataset
df.shape

# There are 1728 instances and 7 variables in the data set.

# preview the top of the dataset
df.head()

# rename column names
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names
col_names

# preview the end of the dataset
df.tail()

# We can see that the column names are renamed. Now, the columns have meaningful names with "**class**" as the target variable.

# ## Inspect dataset

df.info()

for col in col_names:    
    print(df[col].value_counts())
    print ("\n")

# ## Declare feature vector and target variable
#

X = df.drop(['class'], axis=1)
y = df['class']

# ## Split data into separate training and test set

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
X_train.shape, X_test.shape

# ## Feature Engineering
#
# **Feature Engineering** is the process of transforming raw data into useful features that help us to understand our model better and increase its predictive power.

# encode variables with ordinal encoding
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

X_train.head()

X_test.head()

# ## Decision Tree Classifier with criterion gini index

# create classifier
md = 3
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth = md, random_state=0)

# train classifier and predict results
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)
acc = accuracy_score(y_test, y_pred_gini)

# + tags=[]
# decision-tree visualization
plt.figure(figsize=(12,8))
tree.plot_tree(clf_gini.fit(X_train, y_train))
plt.savefig("tree.jpg")
# -

# ## MLFlow code

# + jupyter={"source_hidden": true} tags=[]
## MLFlow snippet
#import mlflow
#from urllib.parse import urlparse
#
## with mlflow.start_run():
#mlflow.end_run()
#mlflow.start_run()
#
#tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
## Model registry does not work with file store
#if tracking_url_type_store != "file":
## Register the model
## There are other ways to use the Model Registry, which depends on the use case,
## please refer to the doc for more information:
## https://mlflow.org/docs/latest/model-registry.html#api-workflow
#    mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
#else:
#    mlflow.sklearn.log_model(lr, "model")
#
#mlflow.end_run()
#---------------------------------------------------------------------
#logged_model = 'runs:/XXXXX/model'
#loaded_model = mlflow.pyfunc.load_model(logged_model)
#
#y_pred_gini = clf_gini.predict(X_test)
#print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))

# + jupyter={"source_hidden": true} tags=[]
## MLFlow snippet
## locally log the classifier
#with mlflow.start_run():
#    mlflow.log_param("max_depth", md)
#    mlflow.log_metric("accuracy", acc)
#    mlflow.sklearn.log_model(clf_gini, "model")

# +
# register the classifier
mlflow.set_tracking_uri(registry_uri)
mlflow.set_experiment('TreeClassifier')

with mlflow.start_run(run_name='blade_runner'):
    mlflow.log_param("max_depth", md)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(sk_model=clf_gini, artifact_path='', registered_model_name='tree_model')
    mlflow.log_artifact("tree.jpg", artifact_path='plots')
# -

# compare accuracy vs. training prediction
print('Model accuracy score with criterion gini index: {0:0.4f}'. format(acc))
y_pred_train_gini = clf_gini.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'. format(acc))

# Here, the training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021. These two values are quite comparable. So, there is no sign of overfitting. 

# ## Decision Tree Classifier with criterion entropy

# train classifier
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf_en.fit(X_train, y_train)

y_pred_en = clf_en.predict(X_test)
print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))

# We can see that the training-set score and test-set score is same as above. The training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021. These two values are quite comparable. So, there is no sign of overfitting. 
#

plt.figure(figsize=(12,8))
tree.plot_tree(clf_en.fit(X_train, y_train))

# + [markdown] tags=[]
# ## Confusion matrix
#
#
# A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.
#
#
# Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-
#
#
# **True Positives (TP)** – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.
#
#
# **True Negatives (TN)** – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.
#
#
# **False Positives (FP)** – False Positives occur when we predict an observation belongs to a    certain class but the observation actually does not belong to that class. This type of error is called **Type I error.**
#
#
#
# **False Negatives (FN)** – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called **Type II error.**
#
#
#
# These four outcomes are summarized in a confusion matrix given below.
#
# -

# print the Confusion Matrix and slice it into four pieces
cm = confusion_matrix(y_test, y_pred_en)
print('Confusion matrix\n\n', cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_en.classes_)
disp.plot()
#plt.show()

# ## Classification report
#
# **Classification report** is another way to evaluate the classification model performance. It displays the  **precision**, **recall**, **f1** and **support** scores for the model. I have described these terms in later.
#
# We can print a classification report as follows:

print(classification_report(y_test, y_pred_en))

# ## References
#
# The work done in this project is inspired from following books and websites:-
#
# 1. Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron
#
# 2. Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido
#
# 3. https://en.wikipedia.org/wiki/Decision_tree
#
# 4. https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
#
# 5. https://en.wikipedia.org/wiki/Entropy_(information_theory)
#
# 6. https://www.datacamp.com/community/tutorials/decision-tree-classification-python
#
# 7. https://stackabuse.com/decision-trees-in-python-with-scikit-learn/
#
# 8. https://acadgild.com/blog/decision-tree-python-code
#
