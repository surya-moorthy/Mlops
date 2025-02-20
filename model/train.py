from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

data = pd.read_csv("house.csv")
data.dropna(inplace=True)

x , y = data.drop(columns=["MEDV"]) , data["MEDV"]

x_train , x_test , y_train , y_test =  train_test_split(x,y,test_size=0.2,random_state=42)

forest_params = [{'max_depth': list(range(10, 15)), 'max_features': list(range(1,10))}]

clf = GridSearchCV(RandomForestRegressor(), forest_params, cv = 5, scoring='neg_mean_squared_error')

clf.fit(x_train, y_train)

print(clf.best_params_)
print(clf.best_score_)