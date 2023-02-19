"""Climate predictor - the data prediction part
"""

from load_data import import_inputs, import_targets, visualize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    """Our cool neural network for predicting climate."""
    def __init__(self) -> None:
        """Initialize the neural network."""
        super(Net, self).__init__()
        self.f = nn.Linear(64, 64)
        self.g = nn.Linear(64, 64)
        self.h = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass on the input x."""
        x = F.relu(self.f(x))
        x = F.relu(self.g(x))
        x = self.h(x)
        return x


def to_tensor(m: np.ndarray) -> torch.Tensor:
    """Convert a numpy array map as a tensor."""
    return torch.from_numpy(m).float()


def to_array(m: torch.Tensor) -> torch.Tensor:
    """Convert a tensor to a numpy array map."""
    return m.detach().numpy()


def download_inputs(resolution: tuple[int, int] = (360, 180), remove_na: bool = True) -> None:
    """Download all inputs for use in learning."""
    i1 = import_inputs(resolution, False, False, remove_na)
    i2 = import_inputs(resolution, True, True, remove_na)
    inputs = np.r_[i1, i2]

    w, h = resolution
    np.save(f"data/{w}_by_{h}/{'' if remove_na else 'na_'}inputs", inputs)
    print(f"Successfully downloaded inputs in {w} by {h} resolution")


def get_inputs(resolution: tuple[int, int] = (360, 180), remove_na: bool = True) -> np.ndarray:
    """Import all inputs for use in learning."""
    w, h = resolution
    return np.load(f"data/{w}_by_{h}/{'' if remove_na else 'na_'}inputs.npy")


def download_targets(resolution: tuple[int, int] = (360, 180), remove_na: bool = True) -> None:
    """Download all targets for use in learning."""
    i1 = import_targets(resolution, False, False, remove_na)
    i2 = import_targets(resolution, True, True, remove_na)
    targets = np.concatenate([i1, i2], axis=2)

    w, h = resolution
    np.save(f"data/{w}_by_{h}/{'' if remove_na else 'na_'}targets", targets)
    print(f"Successfully downloaded targets in {w} by {h} resolution")


def get_targets(resolution: tuple[int, int] = (360, 180), remove_na: bool = True) -> np.ndarray:
    """Import all targets for use in learning."""
    w, h = resolution
    return np.load(f"data/{w}_by_{h}/{'' if remove_na else 'na_'}targets.npy")


def learn() -> Net:
    """Start the neural network's learning."""
    resolution = (360, 180)
    learning_rate = 0.0000001
    momentum = 0.9
    iterations = 1000

    inputs = get_inputs(resolution)
    targets = get_targets(resolution)
    target = targets[0][0]
    # inputs = inputs[:, :36]

    inputs = Variable(torch.from_numpy(inputs).float())
    target = Variable(torch.from_numpy(target).float().unsqueeze(1))

    net = Net()
    net.train()
    loss_function = nn.MSELoss()

    losses = []
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    for i in range(iterations):
        outputs = net(inputs)
        loss = loss_function(outputs, target)

        print(f"Loss, iteration {i}: {loss.item()}")
        losses.append(loss.item())

        net.zero_grad()
        loss.backward()
        optimizer.step()
    return net

# tar = import_targets(remove_na=False)[0][0].reshape((180, 360))
# inp = import_inputs(remove_na=False)
