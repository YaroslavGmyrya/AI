import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

region_names = {
    1: "N. America",
    2: "S. America",
    3: "Europe",
    4: "Africa",
    5: "Asia",
    6: "Oceania"
}

df = pd.read_csv("dataset.data")

cols = [
    "colours", "red", "green", "blue", "gold", "white", "black", "orange",
    "bars", "stripes",
    "circles", "crosses", "saltires", "quarters", "sunstars", "crescent",
    "triangle", "icon", "animate", "text"
]

X = df[cols]
target = df["landmass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, target, test_size=0.25, random_state=42, stratify=target
)

K = 5

model = Pipeline(steps=[
    ("scaler", StandardScaler()),           
    ("knn", KNeighborsClassifier(n_neighbors=K))
])

model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

countries = ["Vietnam","Western-Samoa","Yugoslavia","Zaire","Zambia","Zimbabwe"]

new_flags = pd.DataFrame([

{
    "colours": 2, "red": 1, "green": 0, "blue": 0, "gold": 1, "white": 0, "black": 0, "orange": 0,
    "bars": 0, "stripes": 0,
    "circles": 0, "crosses": 0, "saltires": 0, "quarters": 0,
    "sunstars": 1, "crescent": 0,
    "triangle": 0, "icon": 0, "animate": 0, "text": 0
},

{
    "colours": 3, "red": 1, "green": 0, "blue": 1, "gold": 0, "white": 1, "black": 0, "orange": 0,
    "bars": 0, "stripes": 0,
    "circles": 0, "crosses": 0, "saltires": 0, "quarters": 1,
    "sunstars": 5, "crescent": 0,
    "triangle": 0, "icon": 0, "animate": 0, "text": 0
},

{
    "colours": 4, "red": 1, "green": 0, "blue": 1, "gold": 1, "white": 1, "black": 0, "orange": 0,
    "bars": 0, "stripes": 3,
    "circles": 0, "crosses": 0, "saltires": 0, "quarters": 0,
    "sunstars": 1, "crescent": 0,
    "triangle": 0, "icon": 0, "animate": 0, "text": 0
},

{
    "colours": 4, "red": 1, "green": 1, "blue": 0, "gold": 1, "white": 0, "black": 0, "orange": 1,
    "bars": 0, "stripes": 0,
    "circles": 1, "crosses": 0, "saltires": 0, "quarters": 0,
    "sunstars": 0, "crescent": 0,
    "triangle": 0, "icon": 1, "animate": 1, "text": 0
},

{
    "colours": 4, "red": 1, "green": 1, "blue": 0, "gold": 0, "white": 0, "black": 1, "orange": 1,
    "bars": 3, "stripes": 0,
    "circles": 0, "crosses": 0, "saltires": 0, "quarters": 0,
    "sunstars": 0, "crescent": 0,
    "triangle": 0, "icon": 0, "animate": 1, "text": 0
},

{
    "colours": 5, "red": 1, "green": 1, "blue": 0, "gold": 1, "white": 1, "black": 1, "orange": 0,
    "bars": 0, "stripes": 7,
    "circles": 0, "crosses": 0, "saltires": 0, "quarters": 0,
    "sunstars": 1, "crescent": 0,
    "triangle": 1, "icon": 1, "animate": 1, "text": 0
}

])

predictions = model.predict(new_flags)

for i, p in enumerate(predictions, 1):
    print(f"Flag {countries[i-1]} → Predicted landmass: {region_names[p]}")