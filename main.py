import numpy
import torch
import torch.nn as nn

in_channels, out_channels, kernel_size = 3, 7, 12


class FcNet(nn.Module):
    def __init__(self):
        super(FcNet, self).__init__()
        self.fc1 = nn.Linear(in_channels*kernel_size*kernel_size, out_channels)

    def forward(self, x):
        out = self.fc1(x)
        return out


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        out = out.reshape(-1)
        return out


def save_tensor(tensor, filename):
    torch.save(tensor, filename)


def load_tensor(filename):
    loaded_tensor = torch.load(filename, weights_only=False)
    return loaded_tensor


def generate_data(number_of_inputs):
    x = torch.randn(number_of_inputs)
    return x


def main():
    lin_model = FcNet()
    # save_tensor(lin_model.fc1.weight, 'fc_weights.pt')
    # save_tensor(lin_model.fc1.bias, 'fc_bias.pt')
    lin_model.fc1.weight = load_tensor('fc_weights.pt')
    lin_model.fc1.bias = load_tensor('fc_bias.pt')
    inputs = generate_data(in_channels*kernel_size*kernel_size)
    conv_inputs = inputs.reshape(in_channels, kernel_size, kernel_size)

    conv_weights_list = []
    for w in lin_model.fc1.weight:
        conv_w = w.reshape(in_channels, kernel_size, kernel_size)
        conv_weights_list.append(conv_w)
    conv_weights = torch.stack(conv_weights_list)

    conv_model = ConvNet()
    conv_model.conv1.weight = nn.Parameter(conv_weights)
    conv_model.conv1.bias = nn.Parameter(lin_model.fc1.bias)

    print(lin_model(inputs))
    print(conv_model(conv_inputs))


if __name__ == '__main__':
    main()
