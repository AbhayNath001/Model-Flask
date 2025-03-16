import pandas as pd

# Load Excel dataset
df = pd.read_excel("Final Dataset.xlsx", usecols=[1])

# Save as JSON
df.to_json("Final_Dataset.json", orient="records", lines=True)