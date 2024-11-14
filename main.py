import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class FcNet(nn.Module):
    def __init__(self, input_size, input_fc2_size, out_size):
        super(FcNet, self).__init__()
        self.fc1 = nn.Linear(input_size, input_fc2_size)
        self.fc2 = nn.Linear(input_fc2_size, out_size)
        self.fc3 = nn.Linear(out_size, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, kernel_size):
        super(ConvNet, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, kernel_size
        self.fc1 = nn.Linear(input_size, in_channels * kernel_size * kernel_size)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1)
        self.fc3 = nn.Linear(out_channels, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = out.reshape(self.in_channels, self.kernel_size, self.kernel_size)
        out = self.conv2(out)
        out = out.reshape(-1)
        out = F.relu(out)
        out = self.fc3(out)
        return out


def save_model(model, sample_input: torch.Tensor, model_path: str) -> None:
    torch.save(model.state_dict(), model_path + '.ckpt')
    torch.onnx.export(
        model,  # The model to be exported
        sample_input,  # The sample input tensor
        model_path + ".onnx",  # The output file name
        export_params=True,  # Store the trained parameter weights inside the model file
        opset_version=17,  # The ONNX version to export the model to
        do_constant_folding=True,  # Whether to execute constant folding for optimization
        input_names=['input'],  # The model's input names
        output_names=['output'],  # The model's output names
    )


def load_model(model, model_path: str) -> None:
    model.load_state_dict(torch.load(model_path, weights_only=False))


def load_tensor(filename: str) -> torch.Tensor:
    return torch.load(filename, weights_only=False)


def generate_data(number_of_inputs: int) -> torch.Tensor:
    x = torch.randn(number_of_inputs)
    for i in range(number_of_inputs):
        x[i] *= 100
    return x


def change_conv_weights(fc_model: FcNet, conv_model: ConvNet, in_channels: int, kernel_size: int) -> None:
    conv_model.fc1.weight = fc_model.fc1.weight
    conv_model.fc3.weight = fc_model.fc3.weight
    conv_model.fc1.bias = fc_model.fc1.bias
    conv_model.fc3.bias = fc_model.fc3.bias

    conv_weights_list = []
    for w in fc_model.fc2.weight:
        conv_w = w.reshape(in_channels, kernel_size, kernel_size)
        conv_weights_list.append(conv_w)
    conv_weights = torch.stack(conv_weights_list)

    conv_model.conv2.weight = nn.Parameter(conv_weights)
    conv_model.conv2.bias = fc_model.fc2.bias


def change_weights(input_size: int,in_channels: int, out_channels: int, kernel_size: int) -> None:
    fc_model = FcNet(input_size, in_channels * kernel_size * kernel_size, out_channels)
    conv_model = ConvNet(input_size, in_channels, out_channels, kernel_size)
    change_conv_weights(fc_model, conv_model, in_channels, kernel_size)

    inputs = load_tensor('inputs.pt')

    save_model(fc_model, inputs, 'model/fc_model')
    save_model(conv_model, inputs, 'model/conv_model')


def compare_models(input_size: int, in_channels: int, out_channels: int, kernel_size: int) -> None:
    fc_model = FcNet(input_size, in_channels * kernel_size * kernel_size, out_channels)
    conv_model = ConvNet(input_size, in_channels, out_channels, kernel_size)
    load_model(fc_model, 'model/fc_model.ckpt')
    load_model(conv_model, 'model/conv_model.ckpt')

    inputs = load_tensor('inputs.pt')
    print(f'Выход с fc слоя: {fc_model(inputs)}')
    print(f'Выход с conv слоя: {conv_model(inputs)}')


def generate_input(input_size: int) -> None:
    inputs = generate_data(input_size)
    torch.save(inputs, 'inputs.pt')


def main():
    args = sys.argv
    command = args[1:]

    in_channels, out_channels, kernel_size = 3, 7, 12
    input_size = 3

    try:
        if command[0] == 'change_weights':
            change_weights(input_size, in_channels, out_channels, kernel_size)
        elif command[0] == 'compare_models':
            compare_models(input_size, in_channels, out_channels, kernel_size)
        elif command[0] == 'new_inputs':
            generate_input(input_size)
        else:
            raise Exception()
    except Exception:
        print('An error has occurred')


if __name__ == '__main__':
    main()
