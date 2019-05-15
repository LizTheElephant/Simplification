class ConvNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

    def forward(self, x):

        x = x.view(x.shape[0], -1)

        return x