import pandas as pd
import matplotlib.pyplot as plt

data_description = "data/results.csv"

df = pd.read_csv(data_description, sep=";")


print(df.groupby("Gender").count())

gender = df.groupby("Gender").count()["CID"].to_numpy()
plt.pie(gender, labels=["female", "male"], autopct='%.1f%%')
plt.title("Gender")
plt.show()

print(df.groupby("CID").count()["Image"].to_string())

print(df.groupby("Ethnicity").count())


gender = df.groupby("Ethnicity").count()["CID"].to_numpy()
plt.pie(gender, labels=["caucasian", "asian", "south asian", "black", "middle eastern", "hispanic"], autopct='%.1f%%')
plt.title("Ethnicity")
plt.show()

print(df.groupby("LR").count())


gender = df.groupby("LR").count()["CID"].to_numpy()
plt.pie(gender, labels=["left", "right"], autopct='%.1f%%')
plt.title("Ear orientation")
plt.show()
