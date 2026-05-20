import pandas as pd

df=pd.read_csv("josaa_final_dataset.csv")

#first five rows
print("\n head")
print(df.head())


#shape

print("\n")
print(df.shape)

#data set info
print("\n info of dat is")
print(df.info())

#counting college types
print("\n college type is as follows:")
print(df["College_Type"].value_counts())