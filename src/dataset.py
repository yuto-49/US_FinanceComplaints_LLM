#import torch
#from torch.utils.data import Dataset
import pandas as pd

#class CustomDataset(Dataset):
#    def __init__(self, csv_file, transform=None):
#        self.data = pd.read_csv(csv_file)
#        self.transform = transform

#    def __len__(self):
#        return len(self.data)

#    def __getitem__(self, idx):
#        features = self.data.iloc[idx, :-1].values.astype(float)
#        label = self.data.iloc[idx, -1]
#        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
df = pd.read_csv("./data/consumer_complaints.csv", low_memory=False)
# Check if the column exists
if "consumer_complaint_narrative" not in df.columns:
    raise ValueError("Column 'Consumer complaint narrative' not found.")
# Drop NaN values and convert to string
df["consumer_complaint_narrative"] = df["consumer_complaint_narrative"].dropna().astype(str)
# Convert the DataFrame to a list of strings
texts = df["consumer_complaint_narrative"].tolist()
# Check if the texts are loaded correctly
if not texts:
    raise ValueError("No text data found in 'consumer_complaint_narrative' column.")

# Show a preview to verify it's loaded correctly
print(df.head())
