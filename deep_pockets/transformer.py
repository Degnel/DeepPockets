import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.transformer.attention import MultiHeadAttention
import torch.nn as nn
import torch
from torch import optim

class Transformer(nn.Module):
    """
    Implements a Transformer model with configurable depth.

    Args:
        d_model (int): The dimensionality of the input embeddings.
        n_heads (int): The number of attention heads.
        d_ff (int): The dimensionality of the feedforward network.
        depth (int): The number of encoder layers in the Transformer.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        vocab_size (int, optional): The size of the input vocabulary. If None, no embedding layer is added. Defaults to None.
        max_context_size (int, optional): The maximum length of the input sequences. Defaults to 512.
        mask (bool, optional): If True, adds a mask to the attention scores. Defaults to True.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        depth: int,
        dropout: float = 0.1,
        vocab_size: int | None = None,
        max_context_size: int = 512,
        mask: bool = True,
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.mask = mask

        # Liste des couches de l'encodeur
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                )
                for _ in range(depth)
            ]
        )

        if vocab_size:
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
            self.output_projection.weight = self.embedding.weight
            self.position_embedding = nn.Embedding(max_context_size, d_model)
        else:
            self.embedding = None
            self.output_projection = None
            self.position_embedding = None

    def forward(self, x: torch.Tensor):
        """
        x : Tensor de taille (batch_size, seq_len) ou (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        if self.mask:
            mask = torch.triu(torch.ones(seq_len, seq_len)).bool().to(x.device)
        else:
            mask = None

        if self.embedding is not None:
            x = x.to(torch.int32)
            x = self.embedding(x) + self.position_embedding.weight  # [batch_size, seq_len, d_model]

        # Passage par les couches de l'encodeur
        for layer in self.encoder_layers:
            x = layer(x, mask)

        # Projection finale si applicable
        if self.output_projection is not None:
            x = x - self.position_embedding.weight
            x = self.output_projection(x)

        return x.transpose(1, 2)

    def train_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 20,
        mini_batch_count: int = -1,
        lr: float = 0.001,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        optimizer: optim.Optimizer = optim.AdamW,
        grad_clamp: float = 1,
    ) -> None:
        """
        Train a model to minimize the loss between predicted and target outputs.
        
        Parameters:
        - dataloader (torch.utils.data.DataLoader): Target tensors.
        - epochs (int): Number of training epochs.
        - mini_batch_count (int): Number of training mini_batch
        - lr (float): Learning rate for gradient updates. Defaults to 0.001.
        - criterion (nn.Module): Loss function.
        - optimizer (optim.Optimizer): Optimizer for gradient updates.
        - grad_clamp (float): Maximum gradient value for clipping.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train()
        self.to(device)
        optimizer = optimizer(self.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, (mini_batch, target) in enumerate(dataloader):
                mini_batch, target = mini_batch.to(device), target.to(device)
                optimizer.zero_grad()
                output = self(mini_batch)
                loss = criterion(output, target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), grad_clamp)
                optimizer.step()
                running_loss += loss.item()
                if i + 1 == mini_batch_count:
                    break

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / (i + 1)}")

    def test_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        mini_batch_count: int = -1,
    ) -> float:
        """
        Test a model on a given dataset.

        Parameters:
        - model (nn.Module): The model to train.
        - criterion (nn.Module): Loss function.
        - dataloader (torch.utils.data.Dataloader): Input tensors.
        - mini_batch_count (int): Number of mini_batches to test on.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        self.to(device)
        loss = 0
        for i, (mini_batch, target) in enumerate(dataloader):
            mini_batch, target = mini_batch.to(device), target.to(device)
            output = self(mini_batch)
            loss += criterion(output, target)
            if i + 1 == mini_batch_count:
                break
        loss /= i + 1
        print(f"Score on the whole set, loss: {loss}")
        return loss.item()

class TransformerEncoderLayer(nn.Module):
    """
    Implements a single layer of a Transformer encoder.

    Args:
        d_model (int): The dimensionality of the input embeddings.
        n_heads (int): The number of attention heads.
        d_ff (int): The dimensionality of the feedforward network.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)

        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)

        self.activation = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None):
        """
        x : Tensor de taille (batch_size, seq_len, d_model)
        mask : Tensor de taille (seq_len, seq_len)
        """
        # Attention multi-têtes
        attn_output = self.self_attention(x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        # x = F.normalize(x + self.dropout(attn_output), p=2, dim=-1)

        # Réseau feedforward
        ff_output = self.fc_2(self.activation(self.fc_1(x)))
        x = self.layer_norm2(x + self.dropout(ff_output))
        # x = F.normalize(x + self.dropout(ff_output), p=2, dim=-1)

        return x
