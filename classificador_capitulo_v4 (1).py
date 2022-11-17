import joblib
import gc
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nfe.utils.func import formatted_confusion_matrix

dados = pd.read_csv("../DADOS/capitulo_ncm/PRODUTOS_NFE_2018_TREINAMENTO_NCM_CAPITULO.txt", sep="|", encoding='latin9')
print(dados.__len__())

print(dados.head())

X = dados.loc[:, 'PROD_XPROD'].values.astype('U')
y = dados.loc[:, 'PROD_NCM_CAPITULO']
sw = dados.loc[:, 'CONTAGEM']

del dados
gc.collect()

vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=300000, strip_accents=None)
logreg = LogisticRegression(n_jobs=5, max_iter=150, solver='lbfgs')

pipeline = Pipeline(steps=[
    ("vec", vectorizer),
    ("logreg", logreg)
])


model = pipeline.fit(X, y,  logreg__sample_weight=sw.values)

print("joblib.dump")

joblib.dump(model, "../models/model_capitulo/logistic_regression_model_v4.joblib")

print("model.predict")
predictions = np.array([], dtype=int)

start = 0
step = 500000
end = step
qtd_steps = (round(len(X) / step + 1))
for stp in range(qtd_steps):
    print(f"start: {start}, end: {end}")
    predictions = np.append(predictions, model.predict(X[start:end]))
    start += step
    end += step

print(predictions.__len__())
print(y.__len__())

accuracy = metrics.accuracy_score(y, predictions, sample_weight=sw.values)
print(f"Accuracy: {accuracy}")

classification_report = metrics.classification_report(y, predictions, sample_weight=sw.values, digits=3)
print(classification_report)
print(formatted_confusion_matrix(model, y, predictions, sw.values))
