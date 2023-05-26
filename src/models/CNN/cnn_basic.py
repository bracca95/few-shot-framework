from torch import nn


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.seq2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        out = self.seq1(x)
        out = self.seq2(out)

        return out.view(out.size(0), -1)
    

class CNNCompare(CNN):

    def __init__(self, out_feat: int):
        super().__init__()
        self.out_feat = out_feat

        self.fc1 = nn.Sequential(
            nn.Linear(32 * 7 * 7, self.out_feat),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.seq1(x)
        out = self.seq2(out)

        out = out.view(out.size(0), -1)
        return self.fc1(out)
