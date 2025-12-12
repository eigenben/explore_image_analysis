import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class MnistBasicCNN(nn.Module):
    def __init__(self):
        super(MnistBasicCNN, self).__init__()
        self.cn1 = nn.Conv2d(1, 16, 3, 1)
        self.cn2 = nn.Conv2d(16, 32, 3, 1)
        self.dp1 = nn.Dropout2d(0.10)
        self.dp2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(4608, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.cn1(x)
        x = F.relu(x)
        x = self.cn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MnistBasicCNNRunner:
    def __init__(
        self,
        lr=0.3,
        epochs=3,
        log_every_epochs=1,
    ):
        self.lr = lr
        self.epochs = epochs
        self.log_every_epochs = log_every_epochs
        self._setup_data()
        self.model = MnistBasicCNN().to(device)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        pass

    def _setup_data(self):
        self.train_dataloader = DataLoader(
            datasets.MNIST(
                "./.cache/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1302,), (0.3069,))]
                ),
            ),
            batch_size=32,
            shuffle=True,
        )

        self.test_dataloader = DataLoader(
            datasets.MNIST(
                "./.cache/mnist",
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1302,), (0.3069,))]
                ),
            ),
            batch_size=500,
            shuffle=False,
        )

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for b_i, (x, y) in enumerate(self.train_dataloader):
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                pred_prob = self.model(x)
                loss = F.nll_loss(pred_prob, y)
                loss.backward()
                self.optimizer.step()
                if b_i % 40 == 0:
                    print(
                        f"Train Epoch: {epoch} [{b_i * len(x)}/{len(self.train_dataloader.dataset)}"
                        f" ({100.0 * b_i / len(self.train_dataloader):.0f}%)]\tLoss: {loss.item():.6f}"
                    )

    def test(self):
        self.model.eval()
        loss = 0
        success = 0
        with torch.no_grad():
            for x, y in self.test_dataloader:
                x, y = x.to(device), y.to(device)
                pred_prob = self.model(x)
                loss += F.nll_loss(pred_prob, y, reduction="sum").item()
                pred = pred_prob.argmax(dim=1, keepdim=True)
                success += pred.eq(y.view_as(pred)).sum().item()
        loss /= len(self.test_dataloader.dataset)
        accuracy = 100.0 * success / len(self.test_dataloader.dataset)
        return {
            "loss": loss,
            "accuracy": accuracy,
            "success": success,
            "total": len(self.test_dataloader.dataset),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize predictions on test samples"
    )
    args = parser.parse_args()

    runner = MnistBasicCNNRunner(epochs=3)
    runner.train()
    print(runner.test())

    if args.visualize:
        print("\n--- Visualizing Predictions ---")
        test_samples = enumerate(runner.test_dataloader)

        for n in range(5):
            b_i, (sample_data, sample_targets) = next(test_samples)
            plt.imshow(sample_data[0][0], cmap="gray", interpolation="none")
            plt.show()

            print(
                f"Model prediction is: {runner.model(sample_data.to(device)).argmax(dim=1)[0].item()}"
            )
            print(f"Ground truth label is: {sample_targets[0].item()}\n")
