from __future__ import annotations

import torch
from torch import nn


class RecurrentForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 48,
        num_layers: int = 2,
        cell_type: str = "rnn",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        recurrent_cls: type[nn.Module]
        normalized_type = cell_type.lower()
        if normalized_type == "rnn":
            recurrent_cls = nn.RNN
        elif normalized_type == "lstm":
            recurrent_cls = nn.LSTM
        elif normalized_type == "gru":
            recurrent_cls = nn.GRU
        else:
            raise ValueError(f"Unsupported cell_type: {cell_type}")

        effective_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = recurrent_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        recurrent_output, _ = self.recurrent(inputs)
        final_state = recurrent_output[:, -1, :]
        return self.output(final_state)
