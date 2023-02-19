"""Climate predictor - the temperature prediction part

"""

from process_data import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch


class TemperatureNet(nn.Module):
    """Our cool neural network for predicting 12-month temperature."""

    def __init__(self) -> None:
        """Initialize the neural network."""
        super(TemperatureNet, self).__init__()
        self.f = nn.Linear(110, 110)
        self.g = nn.Linear(110, 12)

        # self.a = nn.Linear(1, 1)
        # self.b = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass on the input x."""
        x = F.relu(self.f(x))
        x = self.g(x)
        return x

    def randomize_weights(self) -> None:
        """Randomize parameter weights."""
        torch.nn.init.xavier_uniform_(self.f.weight, gain=2.0)
        torch.nn.init.xavier_uniform_(self.g.weight, gain=2.0)


def temp_offset(resolution: tuple[int, int] = (360, 180), learning: bool = True,
                flipped_lr: bool = False, flipped_ud: bool = False) -> np.ndarray:
    """Return an offset prediction."""
    if learning:
        latitude = get_inputs(resolution, True, (False, False, True))[:, 0]
        elevation = get_inputs(resolution, True, (False, False, True))[:, 1]
    else:
        latitude = get_inputs(resolution, False, (flipped_lr, flipped_ud, False))[:, 0]
        elevation = get_inputs(resolution, False, (flipped_lr, flipped_ud, False))[:, 1]

    month = np.repeat(np.arange(0., 12.).reshape(1, 12), latitude.shape, axis=0)
    lat_offset = 70 * np.cos(latitude * np.pi / 180) - 40 - 4 / (np.e ** (latitude ** 2 / 400))
    elev_offset = -19 * elevation / 3000 + 3
    month_offset = -np.repeat(latitude.reshape(latitude.shape[0], 1), 12, axis=1) * np.cos(np.pi * (month - 0.25) / 6) / 6

    lat_offset = np.repeat(lat_offset.reshape(1, lat_offset.shape[0]), 12, axis=0).T
    elev_offset = np.repeat(elev_offset.reshape(1, elev_offset.shape[0]), 12, axis=0).T
    return lat_offset + elev_offset + month_offset


def learn() -> TemperatureNet:
    """Start the neural network's learning."""
    torch.manual_seed(29844)
    resolution = (360, 180)

    inputs = get_temp_inputs(resolution, True, (False, False, True))
    targets = get_targets(resolution, True, (False, False, True))
    target = targets[0].T

    target = target - temp_offset(resolution)

    net = TemperatureNet()
    net.randomize_weights()
    net.train()

    inputs, target = to_tensor(inputs), to_tensor(target)
    losses = []
    boost_losses = []
    offset = 0
    offset = gradient_descent(net, inputs, target, 0.001, 0.9, 500, offset, 10000, True,
                              losses, boost_losses)

    return net


def gradient_descent(net: nn.Module, inputs: torch.Tensor, target: torch.Tensor,
                     learning_rate: float, momentum: float, iterations: int,
                     offset: int = 0, batch_size: Optional[int] = None, boosting: bool = False,
                     losses: Optional[list] = None, boost_losses: Optional[list] = None) -> int:
    """Perform gradient descent.

    :param net: The neural network
    :param inputs: The input matrix
    :param target: The target matrix
    :param learning_rate: The neural network's learning rate
    :param momentum: The neural network's momentum factor
    :param iterations: How many iterations/epochs
    :param offset: The current iteration number (for keeping track of it)
    :param batch_size: Number of input-target combinations to learn from each epoch
    :param boosting: Whether to implement boosting
    :param losses: List of all losses, for storage
    :param boost_losses: List of all boosting losses, for storage
    """
    # latitude = to_tensor(np.repeat(inputs[:, 0].reshape(inputs.shape[0], 1), 12, axis=1))
    # seasonal = torch.tensor([np.cos(np.pi * (i - 0.25) / 6) for i in range(0, 12)]).unsqueeze(1)
    # one = torch.tensor([1.])
    # inputs = inputs[:, 4:]

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    # loss_function = torch.nn.MSELoss()

    for i in range(iterations):
        optimizer.zero_grad()

        # latitude_effect = net.a(one) * (0.5 - abs(latitude - net.b(one) * seasonal.T))
        # outputs = net(inputs) + latitude_effect

        # loss = loss_function(outputs, target)
        if batch_size is None:
            outputs = net(inputs)
            loss = torch.norm(outputs - target)
            loss_elementwise = torch.sum(abs(outputs - target), axis=1)
        else:
            batch_indices = np.random.choice(inputs.shape[0], batch_size)
            batch = inputs[batch_indices]
            outputs = net(batch)
            loss = torch.norm(outputs - target[batch_indices])
            loss_elementwise = torch.sum(abs(outputs - target[batch_indices]), axis=1)

        if losses is not None:
            losses.append(loss.item())

        loss.backward()
        optimizer.step()

        # Boosting
        if boosting:
            boost_indices = torch.where(loss_elementwise > torch.median(loss_elementwise))[0]
            boost_inputs = inputs[boost_indices]
            boost_target = target[boost_indices]

            boost_outputs = net(boost_inputs)
            # boost_loss = loss_function(boost_outputs, boost_target)
            boost_loss = torch.norm(boost_outputs - boost_target)

            if boost_losses is not None:
                boost_losses.append(boost_loss.item())

            boost_loss.backward()
            optimizer.step()
            print(f"Loss, iteration {i + offset}: \t {loss.item()} \t Boost: {boost_loss.item()}")
        else:
            print(f"Loss, iteration {i + offset}: \t {loss.item()}")

    return offset + iterations


def predict(net: nn.Module, resolution: tuple[int, int] = (360, 180), flipped_lr: bool = False,
            flipped_ud: bool = False) -> np.ndarray:
    """Return the net prediction for a copy of the Earth."""
    earth = get_temp_inputs(resolution, False, (flipped_lr, flipped_ud, False))
    prediction = to_array(net(to_tensor(earth)))
    return prediction + temp_offset(resolution, False, flipped_ud=flipped_ud, flipped_lr=flipped_lr)


def error(net: nn.Module, resolution: tuple[int, int] = (360, 180), i: int = 0) -> np.ndarray:
    """Return the error between prediction and targets for Earth."""
    prediction = predict(net, resolution)
    targets = get_targets(resolution, False, (False, False, False))[i].T
    return prediction - targets
