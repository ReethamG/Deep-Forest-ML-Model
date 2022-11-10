from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
trainingDataset, testDataset = load_iris(return_X_y=True)
X_train, Y_train, X_test, Y_test = train_test_split(trainingDataset, testDataset, test_size=0.2) #giving me an error
model = RandomForestClassifier(n_estimators=50)
data = pd.DataFrame({'sepallength': iris.data[:, 0], 'sepalwidth': iris.data[:, 1], 'petallength': iris.data[:, 2], 'petalwidth': iris.data[:, 3], 'species': iris.target})
model.fit(X_train, Y_train)

print(model.predict([[3, 3, 2, 2]]))