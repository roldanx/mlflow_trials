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

# # Titanic survival classification problem
# - Download titanic dataset.
# - Train a 2-layer NN with 5 neurons per layer (input/output apart) for XX epochs and 64 batch size.
# - Save the model.

# ## DL model

# + id="a6HRX3PPd69L"
import mlflow
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sklearn.model_selection as ms

from keras import regularizers
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, accuracy_score,recall_score, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
# -

# ## Get MLFlow server URI

registry_uri = os.getenv('REGISTRY_URI')
if not registry_uri:
    raise Exception('REGISTRY_URI env variable should be defined on the system in order to log the generated model')

# ## Prepare dataset

# data load
dataset = sns.load_dataset("titanic")
dataframe, test_dataframe = ms.train_test_split(dataset, train_size=0.7, random_state=1)
dataframe.head(5)

# format data
dataframe = dataframe.astype({"deck": str})
test_dataframe = test_dataframe.astype({"deck": str})

# + colab={"base_uri": "https://localhost:8080/", "height": 643} id="rBLQXDX3e52R" outputId="9583b219-3f54-4440-b39a-465526898d7c" tags=[]
# data processing
for i in dataframe.index:
    if dataframe['deck'][i] == 'nan':
        dataframe.loc[i,'deck'] = 'Z'

for i in test_dataframe.index:
    if test_dataframe['deck'][i] == 'nan':
        test_dataframe.loc[i,'deck'] = 'Z'

train_median = dataframe['age'].median()
for i in dataframe.index:
    if dataframe['age'][i] != dataframe['age'][i]:
        dataframe.loc[i,'age'] = train_median

train_median = test_dataframe['age'].median()
for i in test_dataframe.index:
    if test_dataframe['age'][i] != test_dataframe['age'][i]:
        test_dataframe.loc[i,'age'] = train_median

X = dataframe[['sex', 'pclass', 'age', 'deck']]
y = dataframe[['alive']]
X_ts = test_dataframe[['sex', 'pclass', 'age', 'deck']]
y_ts = test_dataframe[['alive']]
X_ts.head(10)

# + colab={"base_uri": "https://localhost:8080/", "height": 243} id="HByq3PZau4yA" outputId="a965de61-e7c8-4180-935f-9e7d741bbf55"
# normalization
normalization = [X.loc[:, 'age'].mean(), X.loc[:, 'age'].std()]
print("Age normalization --> " + str(normalization))

X.loc[:, 'age'] = (X.loc[:, 'age'] - normalization[0]) / normalization[1]
X_ts.loc[:, 'age'] = (X_ts.loc[:, 'age'] - normalization[0]) / normalization[1]

X_dum = pd.get_dummies(X)
X_ts_dum = pd.get_dummies(X_ts)

# + colab={"base_uri": "https://localhost:8080/"} id="y0mpaCD7F2oU" outputId="9f0ff756-73d7-44f5-e426-8a499d898eb5"
# dummification
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y.values.ravel())
y = integer_encoded.reshape(len(integer_encoded), 1)
integer_encoded_ts = label_encoder.fit_transform(y_ts.values.ravel())
y_ts = integer_encoded_ts.reshape(len(integer_encoded_ts), 1)
# -

# ## Model implementation

# + colab={"base_uri": "https://localhost:8080/"} id="6__EfUNR2S33" outputId="9ecbb7eb-29fc-4569-d8ae-c3c39819949e"
opt = Adam(learning_rate=0.01)

# model definition
input = Input(len(X_dum.columns))
layer_1 = Dense(5, activation='relu')(input)
layer_2 = Dense(5, activation='relu')(layer_1)
output = Dense(1, activation='sigmoid')(layer_2)

model = Model(input, output)
model.compile(loss='binary_crossentropy',
          optimizer=opt,
          metrics=['accuracy'])
model.summary()

# + colab={"base_uri": "https://localhost:8080/"} id="m_RSOAGo-JXG" outputId="eebb7f41-dba0-4754-f8e2-cb1d92026dec"
hist=model.fit(X_dum, 
               y,
               batch_size=64,
               epochs=5,
               validation_split=0.1,
               shuffle=True)

# +
# model.save("titanic_DeepLearn_model")
# -

hist.history.keys()

probabilities = model.predict(X_ts_dum)
fpr, tpr, _ = roc_curve(y_ts, probabilities)
auc = auc(fpr, tpr)
print("Max ROC:")
print(auc)

# +
# register the classifier
mlflow.set_tracking_uri(registry_uri)
mlflow.set_experiment('NeuralNetwork')

with mlflow.start_run(run_name='forest_gump'):
    mlflow.log_metric("auc", auc)
    mlflow.keras.log_model(keras_model=model, artifact_path='', registered_model_name='neural_network')
# -

predictions = np.where(probabilities > .5, 1, 0)
cm = confusion_matrix(y_true=y_ts, y_pred=predictions)

labels = ['Survivor', 'Dead']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()


# +
def plot_curves(history):
  plt.figure()
  plt.xlabel('Épocas')
  plt.ylabel('Error')
  plt.plot(history['loss'])
  plt.plot(history['val_loss'])
  plt.legend(['Entrenamiento', 'Validación'])

  plt.figure()
  plt.xlabel('Épocas')
  plt.ylabel('Accuracy')
  plt.plot(history['accuracy'])
  plt.plot(history['val_accuracy'])
  plt.legend(['Entrenamiento', 'Validación'], loc='lower right')

plot_curves(hist.history)
# -

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
