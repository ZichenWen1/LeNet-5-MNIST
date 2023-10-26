import torch
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np

# Define your test dataset and data loader
test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=256)

# Load the saved model
model = torch.load("./models/mnist_0.984.pkl")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set the model to evaluation mode
model.eval()

all_correct_num = 0
all_sample_num = 0
model.eval()

for idx, (test_x, test_label) in enumerate(test_loader):
    test_x = test_x.to(device)
    test_label = test_label.to(device)
    predict_y = model(test_x.float()).detach()
    predict_y = torch.argmax(predict_y, dim=-1)
    current_correct_num = predict_y == test_label
    all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
    all_sample_num += current_correct_num.shape[0]
acc = all_correct_num / all_sample_num
print('accuracy: {:.3f}'.format(acc), flush=True)