"""1D-CNN and LSTM gesture classifier definitions.

Both architectures operate on landmark sequences of shape (batch, seq_len, features)
and output class logits of shape (batch, num_classes).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GestureCNN1D(nn.Module):
    """1D Convolutional classifier over temporal landmark sequences."""

    def __init__(
        self,
        input_features: int = 63,
        num_classes: int = 4,
        channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        channels = channels or [64, 128, 256]
        kernel_sizes = kernel_sizes or [3, 3, 3]

        layers: list[nn.Module] = []
        in_ch = input_features
        for out_ch, ks in zip(channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ])
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, features).

        Returns:
            Logits of shape (batch, num_classes).
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.features(x)
        return self.classifier(x)


class GestureLSTM(nn.Module):
    """Bidirectional LSTM classifier over temporal landmark sequences."""

    def __init__(
        self,
        input_features: int = 63,
        num_classes: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * direction_factor, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, features).

        Returns:
            Logits of shape (batch, num_classes).
        """
        output, _ = self.lstm(x)
        # Use the last time step's output
        last_output = output[:, -1, :]
        return self.classifier(last_output)
