import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv("dataset_1000_rows.csv")

X = data[["skills_count","experience","domain_match"]]
y = data["result"]

model = LogisticRegression()
model.fit(X,y)

pickle.dump(model,open("model.pkl","wb"))

print("Model trained")