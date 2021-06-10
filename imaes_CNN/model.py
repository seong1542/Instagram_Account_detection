import torch
import torchvision
from torchsummary import summary

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

def cnn_model(name='resnet18', pretrained=True, num_output=2):
    if name == 'resnet18':
        net = torchvision.models.resnet18(pretrained=pretrained).to(DEVICE)
        net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7,7), stride=(2, 2), padding=(1, 1), bias=False)
        net.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.8),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_output))
    elif name == 'mobilenet':
        net = torchvision.models.mobilenet_v2(pretrained=pretrained).to(DEVICE)
        net.features[0][0] = torch.nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        net.classifier = torch.nn.Sequential(
            torch.nn.Linear(1280, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.8),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_output))
    elif name == 'shufflenet':
        net = torchvision.models.shufflenet_v2_x0_5(pretrained=pretrained).to(DEVICE)
        net.conv1[0] = torch.nn.Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        net.fc = torch.nn.Sequential(
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(p=0.8),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_output))

    else:
        print('wrong cnn model!')

    return net

def test():
    net = cnn_model(name='shufflenet').to(DEVICE)

    summary(net, (3, 1400, 640))
    #y = net(torch.randn(1, 3, 480, 640))
    print(net)


if __name__ == "__main__":
    test()
