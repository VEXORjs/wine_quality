import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

red = pd.read_csv("winequality-red.csv", sep=";")
white = pd.read_csv("winequality-white.csv", sep=";")

red['is_red'] = 1
white['is_red'] = 0
df = pd.concat([red, white], ignore_index=True)

X = df.drop('quality', axis=1).values
y = df['quality'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class WineDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels


        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            x_val = self.features[idx]
            y_val = self.labels[idx]

            x_tensor = torch.tensor(x_val, dtype=torch.float32)
            y_tensor = torch.tensor(y_val, dtype=torch.float32)

            return x_tensor, y_tensor


dataset = WineDataset(X_train_scaled, y_train)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class TabularTransformer(nn.Module):
    def __init__(self, num_features=12, embed_dim=64, num_heads=4, dropout_rate=0.2):
        super().__init__()
        self.value_embedding = nn.Linear(1, embed_dim)
        self.feature_embedding = nn.Embedding(num_features, embed_dim)

        encoded_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoded_layer, num_layers=3)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)
        num_features = x.size(1)

        x_val = self.value_embedding(x.unsqueeze(-1))

        feature_ids = torch.arange(num_features).expand(batch_size, -1)
        x_feat = self.feature_embedding(feature_ids)

        x_combined = x_val + x_feat

        transformed_output = self.transformer(x_combined)

        pooled_output = transformed_output.mean(dim=1)

        dropout_output = self.dropout(pooled_output)

        return self.output_layer(dropout_output)

model = TabularTransformer()
adam = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 10
patience = 2
patience_counter = 0
best_val_loss = float("inf")
best_model_weights = None

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

print("Rozpoczynam trenowanie modelu...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in dataloader:
        adam.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions.squeeze(), y_batch)
        loss.backward()
        adam.step()
        total_loss += loss.item()

    train_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, MSE: {train_loss:.4f}")

    model.eval()
    with torch.no_grad():

        test_predictions = model(X_test_tensor)

        test_loss = criterion(test_predictions.squeeze(), y_test_tensor).item()

    if test_loss < best_val_loss:
        best_val_loss = test_loss
        patience_counter = 0
        best_model_weights = copy.deepcopy(model.state_dict())

    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"\n EARLY STOPPING zadziałał w epoce {epoch + 1}!")
        print(f"Brak poprawy od {patience} epok. Przywracam najlepsze wagi.")
        break

if best_model_weights is not None:
    model.load_state_dict(best_model_weights)

print("-" * 30)
print(f"Ostateczny wynik - test MSE: {best_val_loss:.4f}")