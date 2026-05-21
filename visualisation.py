import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("josaa_final_dataset.csv")

plt.hist(df["Closing Rank"])
plt.title("closing rank distibution")
plt.xlabel("Closing Rank")
plt.ylabel("freq")

plt.show()

