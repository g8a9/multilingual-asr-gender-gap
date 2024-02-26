"""
Code utilities adapted from https://github.com/technion-cs-nlp/gender_internal/blob/master/compression/MDLProbingUtils.py
"""

from torch import nn
import torch


def build_probe(input_size, num_classes=2, probe_type="mlp"):
    probes = {
        "mlp": lambda: nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Tanh(),
            nn.Linear(input_size // 2, num_classes),
        ),
        "linear": lambda: nn.Linear(input_size, num_classes),
    }
    return probes[probe_type]()


class MDLProbeTrainer:
    def __init__(
        self,
        input_size: int = 1024,
        num_classes: int = 2,
        probe_type: str = "mlp",
        device: str = "cpu",
        learning_rate: float = 0.0002,
        epochs: int = 3,
    ):
        self.input_size = input_size
        self.num_classes = num_classes
        self.probe_type = probe_type
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs

    def _step(self, X_train, y_train, X_test, y_test):
        """
        Instantiate a probe, train it on the training set, and evaluate it on the test set.
        """
        probe = build_probe(self.input_size, self.num_classes, self.probe_type).to(
            self.device
        )
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.learning_rate)

        # create a training and testing dataloader, iterate over the training set for a few epochs
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_train).float(), torch.tensor(y_train).long()
            ),
            batch_size=32,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_test).float(), torch.tensor(y_test).long()
            ),
            batch_size=32,
            shuffle=False,
        )

        for _ in range(self.epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = probe(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # evaluate the probe on the test set
        probe.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = probe(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                # _, predicted = torch.max(outputs, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()

        return running_loss
        # return correct / total
