import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv(r"C:\Users\jonoz\Downloads\heart.csv")

dataframe = pd.DataFrame(data)

# data plot
plt.matshow(dataframe.corr())
plt.xticks(np.arange(14), dataframe.columns, rotation=90)
plt.yticks(np.arange(14), dataframe.columns, rotation=0)
plt.colorbar()
plt.show()

x = np.asarray(dataframe[['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']])
y = np.asarray(dataframe[['target']])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.6, shuffle= True)

model = RandomForestClassifier(n_estimators=1000, max_features='sqrt', random_state=42)
# model = SVC(verbose = 1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

heart_disease = {0: 'No Heart Disease', 1: 'Possible Heart Disease'}
y_pred_labels = [heart_disease[label] for label in y_pred]

for i in range(len(y_pred_labels)):
    print('Sample', i, ':', y_pred_labels[i])

print(classification_report(y_test, y_pred))

conf = confusion_matrix(y_test, y_pred)

# confusion matrix plot
plt.figure(figsize = (8, 6))
sns.heatmap(conf, annot = True, fmt = 'd', cmap = 'Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()