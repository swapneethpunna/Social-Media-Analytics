import pandas as pd
d = { "name" : ["Ellie","swap"]}#,"year" : ["Junior","senior"], "major" : ["Psycholog","ece"]}
# df=pd.read_csv("statemappings.csv")
df = pd.read_csv("data/statemappings.csv")
print(df)
x=df.values[0]
print(x)