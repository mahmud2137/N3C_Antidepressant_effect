import pandas as pd
import numpy as np


df = pd.read_excel("N3C_data.xlsx")
df.drop(columns=['Unnamed: 0'], inplace=True)