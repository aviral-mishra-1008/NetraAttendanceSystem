import pandas as pd
df = pd.read_csv('Attendance Record\\2026\\CSE\\D\\CSN14401.csv')
rolls = []
for i in df.Roll:
    rolls.append(i)
