# -*- eval: (progn (require 'outli) (outli-mode)); -*-
# * import
import time
import sys
import numpy as np
import torch
import torch.utils
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from concrete.ml.torch.compile import compile_torch_model

# And some helpers for visualization.

# %matplotlib inline

# import matplotlib.pyplot as plt

# * load dataset X, y %%
# X, y = load_digits(return_X_y=True) # load dataset

# # The sklearn Digits data-set, though it contains digit images, keeps these images in vectors
# # so we need to reshape them to 2D first. The images are 8x8 px in size and monochrome
# X = np.expand_dims(X.reshape((-1, 8, 8)), 1)

# nplot = 4
# fig, ax = plt.subplots(nplot, nplot, figsize=(6, 6))
# for i in range(0, nplot):
#     for j in range(0, nplot):
#         ax[i, j].imshow(X[i * nplot + j, ::].squeeze())
# plt.show()

# x_train, x_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, shuffle=True, random_state=42
# )
# * network define #%%

class DeepCNNX(nn.Module):
    """A very small CNN to classify the sklearn digits data-set."""
    # self.conv_layers = nn.ModuleList([qnn.QuantConv2d(92, 92, kernel_size=1, weight_bit_width=6, bias=False) for _ in range(20)])
    # self.conv_last = qnn.QuantConv2d(92, 16, kernel_size=2, weight_bit_width=4, bias=False)
    # self.fc = qnn.QuantLinear(16, 10, weight_bit_width=6, bias=True)

    def __init__(self, x = 20) -> None:
        """Construct the CNN with a configurable number of classes.
        x: number of 1-conv
        """
        super().__init__()

        # This network has a total complexity of 1216 MAC
        self.conv1 = nn.Conv2d(1, 2, 3, stride=1)
        self.conv2 = nn.Conv2d(2, 92, 3, stride=2)
        self.conv_layers = nn.ModuleList([nn.Conv2d(92, 92, kernel_size=1) for _ in range(x)])
        self.conv_last = nn.Conv2d(92, 16, kernel_size=2)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        """Run inference on the tiny CNN, apply the decision layer on the reshaped conv output."""
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        for layer in self.conv_layers:
          x = torch.relu(layer(x))
        x = torch.relu(self.conv_last(x))
        x = x.view(-1, self.num_flat_features(x)) # flatten to (n, feature_num)
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
      # return product(x.size(1:))
      size = x.size()[1:]
      num_features = 1
      for s in size:
        num_features *= s
      return num_features

# * set seet
torch.manual_seed(42)

# * train_one_epoch
def train_one_epoch(net, optimizer, train_loader):
    # Cross Entropy loss for classification when not using a softmax layer in the network
    loss = nn.CrossEntropyLoss()

    net.train()
    avg_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = net(data)
        loss_net = loss(output, target.long())
        loss_net.backward()
        optimizer.step()
        avg_loss += loss_net.item()

    return avg_loss / len(train_loader)

# define net

if len(sys.argv) == 2:
  x = int(sys.argv[1])
else:
  x = 20

net = DeepCNNX(x=x)
# * begin train (commented)
# # Create the tiny CNN with 10 output classes
# N_EPOCHS = 150

# # Create a train data loader
# train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
# train_dataloader = DataLoader(train_dataset, batch_size=64)

# # Create a test data loader to supply batches for network evaluation (test)
# test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
# test_dataloader = DataLoader(test_dataset)

# # Train the network with Adam, output the test set accuracy every epoch
# losses_bits = []
# optimizer = torch.optim.Adam(net.parameters())
# for _ in tqdm(range(N_EPOCHS), desc="Training"):
#     losses_bits.append(train_one_epoch(net, optimizer, train_dataloader))

# fig = plt.figure(figsize=(8, 4))
# plt.plot(losses_bits)
# plt.ylabel("Cross Entropy Loss")
# plt.xlabel("Epoch")
# plt.title("Training set loss during training")
# plt.grid(True)
# plt.show()

# * test_torch
def test_torch(net, test_loader):
    """Test the network: measure accuracy on the test set."""

    # Freeze normalization layers
    net.eval()

    all_y_pred = np.zeros((len(test_loader)), dtype=np.int64)
    all_targets = np.zeros((len(test_loader)), dtype=np.int64)

    # Iterate over the batches
    idx = 0
    for data, target in test_loader:
        # Accumulate the ground truth labels
        endidx = idx + target.shape[0]
        all_targets[idx:endidx] = target.numpy()

        # Run forward and get the predicted class id
        output = net(data).argmax(1).detach().numpy()
        all_y_pred[idx:endidx] = output

        idx += target.shape[0]

    # Print out the accuracy as a percentage
    n_correct = np.sum(all_targets == all_y_pred)
    print(
        f"Test accuracy for fp32 weights and activations: "
        f"{n_correct / len(test_loader) * 100:.2f}%"
    )


# * call test_torch
# test_torch(net, test_dataloader)

# * test_with_concrete
def test_with_concrete(quantized_module, test_loader, use_sim):
    """Test a neural network that is quantized and compiled with Concrete ML."""

    # Casting the inputs into int64 is recommended
    all_y_pred = np.zeros((len(test_loader)), dtype=np.int64)
    all_targets = np.zeros((len(test_loader)), dtype=np.int64)

    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
    idx = 0
    for data, target in tqdm(test_loader):
        data = data.numpy()
        target = target.numpy()

        fhe_mode = "simulate" if use_sim else "execute"

        # Quantize the inputs and cast to appropriate data type
        y_pred = quantized_module.forward(data, fhe=fhe_mode)

        endidx = idx + target.shape[0]

        # Accumulate the ground truth labels
        all_targets[idx:endidx] = target

        # Get the predicted class id and accumulate the predictions
        y_pred = np.argmax(y_pred, axis=1)
        all_y_pred[idx:endidx] = y_pred

        # Update the index
        idx += target.shape[0]

    # Compute and report results
    n_correct = np.sum(all_targets == all_y_pred)

    return n_correct / len(test_loader)

# * compile torch model and forward
n_bits = 4

model_input = np.random.rand(1,1,8,8)
q_module = compile_torch_model(net, model_input, rounding_threshold_bits=n_bits, p_error=0.1)
import time

sim_time = -time.perf_counter()
# accs = test_with_concrete(
#     q_module,
#     test_dataloader,
#     use_sim=True,
# )
fhe_mode = "simulate"
# fhe_mode = "execute"
y_pred = q_module.forward(model_input, fhe=fhe_mode)

sim_time += time.perf_counter()

print(f"Simulated FHE execution for {n_bits} bit network accuracy after {sim_time}")

# # Generate keys first
# t = time.time()
# q_module.fhe_circuit.keygen()
# print(f"Keygen time: {time.time()-t:.2f}s")

# # Run inference in FHE on a single encrypted example
# mini_test_dataset = TensorDataset(torch.Tensor(x_test[:100, :]), torch.Tensor(y_test[:100]))
# mini_test_dataloader = DataLoader(mini_test_dataset)

# t = time.time()
# accuracy_test = test_with_concrete(
#     q_module,
#     mini_test_dataloader,
#     use_sim=False,
# )
# elapsed_time = time.time() - t
# time_per_inference = elapsed_time / len(mini_test_dataset)
# accuracy_percentage = 100 * accuracy_test

# print(
#     f"Time per inference in FHE: {time_per_inference:.2f} "
#     f"with {accuracy_percentage:.2f}% accuracy"
# )