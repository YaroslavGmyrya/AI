import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_file = "dataset.data"

colors = ["green", "red", "gold", "blue", "white", "black", "orange"]
figures = ["circles", "quarters", "crosses", "saltires", "animate", "icon", "text", "sunstars"]

region_names = {
    1: "N. America",
    2: "S. America",
    3: "Europe",
    4: "Africa",
    5: "Asia",
    6: "Oceania"
}

data = pd.read_csv(data_file, sep=",")

data["landmass"] = data["landmass"].map(region_names).fillna("Unknown")

colors_table = data.groupby("landmass")[colors].mean() * 100

sns.heatmap(colors_table, cmap="coolwarm", annot=True)
plt.title("Colors by Region in %")
plt.show() 
 
figures_table = data.groupby("landmass")[figures].mean() * 100

norm_table = (figures_table - figures_table.min()) / (figures_table.max() - figures_table.min()) * 100

sns.heatmap(norm_table, cmap="coolwarm", annot=True)
plt.title("Figures table in %")
plt.show()

cols = [c for c in (colors + figures) if c in data.columns]
corr = data[cols].corr()

sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
plt.title("Corr table")
plt.show()

plt.show()