import torch
from torch import nn


class ProteinCNN(nn.Module):
    """The exact 16,418-parameter CNN from the experiment packet."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(21, 32, kernel_size=7, padding=3, bias=True)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=7, padding=3, bias=True)
        self.classifier = nn.Linear(32 * 70, 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.classifier(x.flatten(start_dim=1))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
