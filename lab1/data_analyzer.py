import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_file = "machine.data"
log_file = "log.txt"

data = pd.read_csv(data_file, sep=",")

info = data.groupby("vendor").agg(
    PRP_mean=("PRP", "mean"),
    PRP_median=("PRP", "median"),
    PRP_min=("PRP", "min"),
    PRP_max=("PRP", "max"),
    PRP_std=("PRP", "std"),

    ERP_mean=("ERP", "mean"),
    ERP_median=("ERP", "median"),
    ERP_min=("ERP", "min"),
    ERP_max=("ERP", "max"),
    ERP_std=("ERP", "std"),

    MYCT_mean=("MYCT", "mean"),
    MYCT_median=("MYCT", "median"),
    MYCT_min=("MYCT", "min"),
    MYCT_max=("MYCT", "max"),

    MMAX_mean=("MMAX", "mean"),
    MMAX_min=("MMAX", "min"),
    MMAX_max=("MMAX", "max"),

    CACH_mean=("CACH", "mean"),
    CACH_max=("CACH", "max"),

    models_count=("Model Name", "count")
)

min_val = min(data["ERP"].min(), data["PRP"].min())
max_val = max(data["ERP"].max(), data["PRP"].max())

plt.scatter(data["ERP"], data["PRP"])
plt.xlabel("ERP")
plt.ylabel("PRP")
plt.title("PRP and ERP compare")

plt.plot([min_val, max_val], [min_val, max_val], color="green")

plt.show()

plt.subplot(2, 1, 1)
plt.bar(info.index, info["PRP_mean"])
plt.xticks(rotation=90)
plt.xlabel("Vendors")
plt.ylabel("Mean PRP")
plt.title("Avg Vendor PRP")

plt.subplot(2, 1, 2)
plt.bar(info.index, info["CACH_mean"])
plt.xticks(rotation=90)
plt.xlabel("Vendors")
plt.ylabel("Mean CACHE")
plt.title("Avg Vendor CACHE")
plt.show()



with open("out.txt", "w") as f:
    f.write(info.to_string())
    
    

