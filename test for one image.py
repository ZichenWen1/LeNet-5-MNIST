import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random

# Load the saved model
model = torch.load("./models/mnist_0.984.pkl")

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Set the model to evaluation mode
model.eval()

# Load the MNIST test dataset
test_dataset = MNIST(root='./test', train=False, transform=ToTensor())

# Choose an image from the dataset (you can change the index as needed)
# index = 5  # Change this index to select a different image

# OR select randomly
index = random.randint(0, len(test_dataset) - 1)

input_image, target = test_dataset[index]
input_image = input_image.to(device)

# Make a prediction
with torch.no_grad():
    input_tensor = input_image.unsqueeze(0)  # Add a batch dimension
    output = model(input_tensor)
    _, predicted = output.max(1)

# Display the chosen image
plt.imshow(input_image.squeeze().cpu().numpy(), cmap='gray')
plt.title(f"True Label: {target}, Predicted Label: {predicted.item()}")
plt.show()
print(f"True Label: {target}, Predicted Label: {predicted.item()}")

