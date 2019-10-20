import torch
import torch.nn as nn

# from torchsummary import summary


class Maml(nn.Module):
    def __init__(self):
        super(Maml, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(64 * 28 * 28, 1024),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(1024, 5),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 64 * 28 * 28)
        x = self.linear_layer(x)
        return x


if __name__ == "__main__":

    maml = Maml()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

