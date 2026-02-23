import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_file = "dataset.data"
log_file = "log.txt"

colors = ["green", "red", "gold", "blue", "white", "black", "orange"]
figures = ["circles", "sunstars", "quarters", "crosses", "saltires", "animate", "icon", "text"]

region_names = {
    1: "N. America",
    2: "S. America",
    3: "Europe",
    4: "Africa",
    5: "Asia",
    6: "Oceania"
}


data = pd.read_csv(data_file, sep=",")

info = data.groupby(["landmass"]).agg(
    #colors
    green=("green", "sum"),
    red=("red", "sum"),
    gold=("gold", "sum"),
    blue=("blue", "sum"),
    white=("white", "sum"),
    black=("black", "sum"),
    orange=("orange", "sum"),
    
    #figure
    circles=("circles", "sum"),
    sunstars=("sunstars", "sum"),
    quarters=("quarters", "sum"),
    crosses=("crosses", "sum"),
    saltires=("saltires", "sum"),
    animate=("animate", "sum"),
    icon=("icon", "sum"),
    
    #text
    text=("text", "sum"),
    
    #count
    count=("country", "count"),
     
)

info.index = info.index.map(region_names)

plt.pie(
    info["count"],
    labels=info.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Country by region distribution")
plt.axis("equal")
plt.savefig("results/country_by_region.png")

for region in info.index:
    values = info.loc[region, colors]
    values2 = info.loc[region, figures]
    plt.figure(facecolor="lightgray")
    plt.subplot(1,2,1)
    plt.pie(
        values,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": "purple"}
    )
    
    plt.title(f"Flag colors in {region}")
    plt.axis("equal")
    
    plt.subplot(1,2,2)
    
    plt.pie(
        values2,
        labels=figures,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": "purple"}
    )
    
    plt.title(f"Flag figures in {region}")
    plt.axis("equal")
    plt.savefig(f"results/{region}_info.png")
    
print(info)


    
    

