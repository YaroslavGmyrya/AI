import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_file = "dataset.data"
log_file = "log.txt"

colors = ["green", "red", "gold", "blue", "white", "black", "orange"]
figures = ["circles", "quarters", "crosses", "saltires", "animate", "icon", "text"]

region_names = {
    1: "N. America",
    2: "S. America",
    3: "Europe",
    4: "Africa",
    5: "Asia",
    6: "Oceania"
}


data = pd.read_csv(data_file, sep=",")

data["landmass"] = data["landmass"].map(region_names)

data.groupby("landmass")[colors].sum()

grp1 = data.groupby("landmass")[colors].mean() * 100

grp2 = data.groupby("landmass")[figures].mean() * 100


sns.heatmap(grp1, cmap="coolwarm")
plt.show()
sns.heatmap(grp2, cmap="coolwarm")


corr = data.corr(numeric_only=True)

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    square=True
)
plt.title("Correlation Matrix")

plt.show()