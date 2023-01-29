"""Climate predictor - the temperature prediction part

"""

from process_data import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch


class TemperatureNet(nn.Module):
    """Our cool neural network for predicting 12-month temperature."""

    def __init__(self) -> None:
        """Initialize the neural network."""
        super(TemperatureNet, self).__init__()
        self.f = nn.Linear(65, 12)
        self.a = nn.Linear(1, 1)
        self.b = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass on the input x."""
        x = self.f(x)
        return x


def learn() -> TemperatureNet:
    """Start the neural network's learning."""
    resolution = (360, 180)

    inputs = get_temp_inputs(resolution, True, (False, False, True))
    targets = get_targets(resolution, True, (False, False, True))
    # target = targets[0][0].reshape(targets.shape[2], 1)
    target = targets[0].T

    latitude = to_tensor(np.repeat(inputs[:, 0].reshape(inputs.shape[0], 1), 12, axis=1))

    inputs = Variable(torch.from_numpy(inputs).float())
    target = Variable(torch.from_numpy(target).float())

    net = TemperatureNet()
    net.train()

    torch.manual_seed(194853)
    torch.nn.init.xavier_uniform_(net.f.weight, gain=2.0)
    # torch.nn.init.xavier_uniform_(net.g.weight, gain=2.0)
    # torch.nn.init.xavier_uniform_(net.h.weight, gain=2.0)

    losses = []
    # boost_losses = []
    offset = 0
    offset = gradient_descent(net, inputs, target, losses, latitude, 0.001, 0.9, 1000, offset)
    # offset = boosted_gradient_descent(net, inputs, target, losses, boost_losses, 0.001, 0.9, 1000, offset)

    return net


def gradient_descent(net: nn.Module, inputs: torch.Tensor, target: torch.Tensor,
                     losses: list[float], latitude: torch.Tensor, learning_rate: float,
                     momentum: float, iterations: int, offset: int) -> int:
    """Perform gradient descent."""
    seasonal = torch.tensor([np.cos(np.pi * (i - 0.25) / 6) for i in range(0, 12)]).unsqueeze(1)
    one = torch.tensor([1.])

    inputs = inputs[:, 4:]

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    # loss_function = torch.nn.MSELoss()

    for i in range(iterations):
        optimizer.zero_grad()

        latitude_effect = net.a(one) * (0.5 - abs(latitude - net.b(one) * seasonal.T))
        outputs = net(inputs) + latitude_effect
        # loss = loss_function(outputs, target)
        loss = torch.norm(outputs - target)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        print(f"Loss, iteration {i + offset}: \t {loss.item()}")

    return offset + iterations


def boosted_gradient_descent(net: nn.Module, inputs: torch.Tensor, target: torch.Tensor,
                             losses: list[float], boost_losses: list[float], learning_rate: float,
                             momentum: float, iterations: int, offset: int) -> int:
    """Perform gradient descent with boosting."""
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    # loss_function = torch.nn.MSELoss()

    for i in range(iterations):
        optimizer.zero_grad()

        outputs = net(inputs)
        # loss = loss_function(outputs, target)
        loss_elementwise = torch.sum(abs(outputs - target), axis=1)
        loss = torch.norm(outputs - target)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # Boosting
        boost_indices = torch.where(loss_elementwise > torch.median(loss_elementwise))[0]
        boost_inputs = inputs[boost_indices]
        boost_target = target[boost_indices]

        boost_outputs = net(boost_inputs)
        # boost_loss = loss_function(boost_outputs, boost_target)
        boost_loss = torch.norm(boost_outputs - boost_target)

        boost_losses.append(boost_loss.item())
        boost_loss.backward()
        optimizer.step()

        print(f"Loss, iteration {i + offset}: \t {loss.item()}, \t {boost_loss.item()}")

    return offset + iterations
