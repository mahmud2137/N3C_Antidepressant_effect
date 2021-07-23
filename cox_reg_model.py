import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

df = pd.read_excel("trazodone.xlsx")
df.drop(columns=['Unnamed: 0'], inplace=True)
df.conditions = df.conditions.apply(lambda x: str(x).split(','))
df.conditions = df.conditions.apply(lambda x: list(map(int, x)))
unique_conditions = list(set(list(itertools.chain.from_iterable(df.conditions.values))))
unique_conditions.sort()
conditions_matrix = np.zeros((len(df),len(unique_conditions)), dtype=np.bool)