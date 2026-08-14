import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import random


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(100,), activation="relu", dropout=None):
        super().__init__()
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "logistic": nn.Sigmoid(),
        }
        assert activation in activations, f"Unsupported activation: {activation}"
        layers = []
        dims = [input_dim] + list(hidden_layer_sizes)
        for i in range(len(hidden_layer_sizes)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activations[activation])

        if dropout is not None:
            layers.append(nn.Dropout(dropout))  # Adding dropout layer
        layers.append(nn.Linear(dims[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        alpha=0.0001,
        batch_size=64,
        learning_rate_init=0.0005,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        early_stopping=False,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        patience=10,
        max_epochs=200,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
            np.random.seed(random_state)
            random.seed(random_state)

        self.to(device)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate_init,
            betas=(beta_1, beta_2),
            eps=epsilon,
            weight_decay=alpha,
        )
        criterion = nn.MSELoss()

        X_train_tensor = torch.tensor(X_train.values if isinstance(X_train, pd.DataFrame) else X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val.values if isinstance(X_val, pd.DataFrame) else X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        best_model_state = None
        no_improve_epochs = 0

        for epoch in range(max_epochs):
            self.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                preds = self(x)
                loss = criterion(preds.squeeze(), y.squeeze())
                loss.backward()
                optimizer.step()

            self.eval()
            val_losses = []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    preds = self(x)
                    loss = criterion(preds.squeeze(), y.squeeze())
                    val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            print(f"Epoch {epoch+1}: val_loss = {val_loss:.4f}")

            if val_loss + tol < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if early_stopping and no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        return self

    def predict(self, X, batch_size=64, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.eval()
        X_tensor = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        dataset = DataLoader(X_tensor, batch_size=batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for x in dataset:
                x = x.to(device)
                y = self(x)
                preds.append(y.cpu().numpy())
        return np.concatenate(preds, axis=0)


class DualCrossAttention(nn.Module):
    def __init__(self, input_dim_gnn, input_dim_rdkit, hidden_dim, attn_dim):
        super().__init__()
        # self.q_gnn = nn.Linear(input_dim_gnn, hidden_dim)
        # self.k_gnn = nn.Linear(input_dim_gnn, hidden_dim)
        # self.v_gnn = nn.Linear(input_dim_gnn, hidden_dim)

        self.q_fp = nn.Linear(input_dim_rdkit, hidden_dim)
        self.k_fp = nn.Linear(input_dim_rdkit, hidden_dim)
        self.v_fp = nn.Linear(input_dim_rdkit, hidden_dim)

        self.attn_gnn_fp = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)
        self.attn_fp_gnn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)

        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.proj = nn.Linear(hidden_dim * 2, attn_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.hidden_dim = hidden_dim

    def forward(self, h_gnn, h_fp):
        # shape: (B, D)
        # Q_GNN = self.q_gnn(h_gnn)
        # K_GNN = self.k_gnn(h_gnn)
        # V_GNN = self.v_gnn(h_gnn)
        Q_GNN = h_gnn
        K_GNN = h_gnn
        V_GNN = h_gnn

        Q_FP = self.q_fp(h_fp)
        K_FP = self.k_fp(h_fp)
        V_FP = self.v_fp(h_fp)

        # Add 1-length sequence dim (B, 1, D) to match attention shape
        Q_GNN, K_FP, V_FP = [x.unsqueeze(1) for x in [Q_GNN, K_FP, V_FP]]
        Q_FP, K_GNN, V_GNN = [x.unsqueeze(1) for x in [Q_FP, K_GNN, V_GNN]]

        # # Attention 1: GNN queries RDKit
        # attn1 = self.softmax(torch.matmul(Q_GNN, K_FP.transpose(-1, -2)) / self.hidden_dim**0.5)
        # out1 = torch.matmul(attn1, V_FP).squeeze(1)

        # # Attention 2: RDKit queries GNN
        # attn2 = self.softmax(torch.matmul(Q_FP, K_GNN.transpose(-1, -2)) / self.hidden_dim**0.5)
        # out2 = torch.matmul(attn2, V_GNN).squeeze(1)

        # # shape: (B, D)
        # return self.proj(self.ln(torch.cat([out1, out2], dim=-1)))

        out1, _ = self.attn_gnn_fp(Q_GNN, K_FP, V_FP)  # (B, 1, D)
        out2, _ = self.attn_fp_gnn(Q_FP, K_GNN, V_GNN)  # (B, 1, D)
        # (B, 1, D) -> (B, D)
        return self.proj(self.ln(torch.cat([out1.squeeze(1), out2.squeeze(1)], dim=-1)))


class CrossAttnMLPRegressor(nn.Module):
    def __init__(self, gnn_dim, rdkit_dim, hidden_dim, attn_dim=150, mlp_hidden=(128, 64)):
        super().__init__()
        self.attn = DualCrossAttention(gnn_dim, rdkit_dim, hidden_dim, attn_dim)

        # self.alpha = nn.Parameter(torch.ones(attn_dim))
        # self.beta = nn.Parameter(torch.ones(attn_dim))

        # dims = [attn_dim * 2] + list(mlp_hidden) + [1]
        dims = [attn_dim] + list(mlp_hidden) + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, gnn_x, rdkit_x):
        attn_x = self.attn(gnn_x, rdkit_x)
        return self.mlp(attn_x)

    # def forward(self, gnn_x, rdkit_x):
    #     attn_x = self.attn(gnn_x, rdkit_x)  # (B, attn_dim)
    #     concat_x = torch.cat([gnn_x, rdkit_x], dim=-1)  # (B, gnn_dim + rdkit_dim)

    #     fused = attn_x * self.alpha + concat_x * self.beta  # (B, attn_dim)
    #     return self.mlp(fused)

    def fit(
        self,
        X_gnn_train, X_rdkit_train, y_train,
        X_gnn_val, X_rdkit_val, y_val,
        alpha=0.0001,
        batch_size=64,
        learning_rate_init=0.0005,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        early_stopping=True,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        patience=10,
        max_epochs=200,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
            np.random.seed(random_state)
            random.seed(random_state)

        self.to(device)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate_init,
            betas=(beta_1, beta_2),
            eps=epsilon,
            weight_decay=alpha,
        )
        criterion = nn.MSELoss()

        X_gnn_train_tensor = torch.tensor(X_gnn_train.values if isinstance(X_gnn_train, pd.DataFrame) else X_gnn_train, dtype=torch.float32)
        X_rdkit_train_tensor = torch.tensor(X_rdkit_train.values if isinstance(X_rdkit_train, pd.DataFrame) else X_rdkit_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        X_gnn_val_tensor = torch.tensor(X_gnn_val.values if isinstance(X_gnn_val, pd.DataFrame) else X_gnn_val, dtype=torch.float32)
        X_rdkit_val_tensor = torch.tensor(X_rdkit_val.values if isinstance(X_rdkit_val, pd.DataFrame) else X_rdkit_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = TensorDataset(X_gnn_train_tensor, X_rdkit_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_gnn_val_tensor, X_rdkit_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        best_model_state = None
        no_improve_epochs = 0

        for epoch in range(max_epochs):
            self.train()
            for gnn_x, rdkit_x, y in train_loader:
                gnn_x, rdkit_x, y = gnn_x.to(device), rdkit_x.to(device), y.to(device)
                optimizer.zero_grad()
                preds = self(gnn_x, rdkit_x)
                loss = criterion(preds.squeeze(), y.squeeze())
                loss.backward()
                optimizer.step()

            self.eval()
            val_losses = []
            with torch.no_grad():
                for gnn_x, rdkit_x, y in val_loader:
                    gnn_x, rdkit_x, y = gnn_x.to(device), rdkit_x.to(device), y.to(device)
                    preds = self(gnn_x, rdkit_x)
                    loss = criterion(preds.squeeze(), y.squeeze())
                    val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            print(f"Epoch {epoch+1}: val_loss = {val_loss:.4f}")

            if val_loss + tol < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if early_stopping and no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        return self

    def predict(self, X_gnn, X_rdkit, batch_size=64, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.eval()
        X_gnn_tensor = torch.tensor(X_gnn.values if isinstance(X_gnn, pd.DataFrame) else X_gnn, dtype=torch.float32)
        X_rdkit_tensor = torch.tensor(X_rdkit.values if isinstance(X_rdkit, pd.DataFrame) else X_rdkit, dtype=torch.float32)
        dataset = DataLoader(TensorDataset(X_gnn_tensor, X_rdkit_tensor), batch_size=batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for gnn_x, rdkit_x in dataset:
                gnn_x, rdkit_x = gnn_x.to(device), rdkit_x.to(device)
                y = self(gnn_x, rdkit_x)
                preds.append(y.cpu().numpy())

        return np.concatenate(preds, axis=0)
