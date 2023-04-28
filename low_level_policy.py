import torch.nn as nn
import torch

# CANM: Causal Discovery with Cascade Nonlinear Additive Noise Models
# Ruichu Cai, Jie Qiao, Kun Zhang, Zhenjie Zhang, Zhifeng Hao. Causal Discovery with Cascade Nonlinear Additive Noise Models. IJCAI 2019
# https://arxiv.org/abs/1905.09442
# https://github.com/DMIRLAB-Group/CANM
class Low_Level_Policy(nn.Module):
    def __init__(self, N) -> None:
        super(Low_Level_Policy, self).__init__()

        self.fc1 = nn.Conv2d(2, 20, (1, 1))
        # encoder mu
        self.fc21 = nn.Conv2d(20, 12, (1, 1))
        self.fc22 = nn.Conv2d(12, 7, (1, 1))
        self.fc23 = nn.Conv2d(7, N, (1, 1))

        # encoder logvar
        self.fc31 = nn.Conv2d(20, 12, (1, 1))
        self.fc32 = nn.Conv2d(12, 7, (1, 1))
        self.fc33 = nn.Conv2d(7, N, (1, 1))

        # decoder
        self.fc4 = nn.Conv2d(N+1, 10, (1, 1))
        self.fc5 = nn.Conv2d(10, 7, (1, 1))
        self.fc6 = nn.Conv2d(7, 5, (1, 1))
        self.fc7 = nn.Conv2d(5, 1, (1, 1))

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def encode(self, xy):
        h1 = self.relu(self.fc1(xy))

        h21 = self.tanh(self.fc21(h1))
        h22 = self.relu(self.fc22(h21))
        mu = self.fc23(h22)

        h31 = self.tanh(self.fc31(h1))
        h32 = self.relu(self.fc32(h31))
        logvar = self.fc33(h32)

        return mu.permute(2, 3, 0, 1), logvar.permute(2, 3, 0, 1)

    def decode(self, x, z):
        h4 = self.tanh(self.fc4(torch.cat((x, z), 1)))
        h5 = self.relu(self.fc5(h4))
        h6 = self.sigmoid(self.fc6(h5))
        yhat = self.fc7(h6)
        return yhat

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)  # e^(0.5*logvar)=e^(logstd)=std
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, dag, data):
        sz = dag.shape[0]
        a_input = torch.cat([data.repeat(1, sz).view(
            sz*sz, -1), data.repeat(sz, 1)], dim=1).view(sz, sz, 2, -1).transpose(2, 3)
        a_input = a_input.permute(2, 3, 0, 1)

        Z_mu, Z_logvar = self.encode(a_input)
        Z = self.reparameterize(Z_mu, Z_logvar).permute(2, 3, 0, 1)
        X = data.repeat(1, sz).view(sz, sz, -1)
        X = X.unsqueeze(-1).permute(2, 3, 0, 1)

        yhat = self.decode(X, Z).permute(2, 3, 0, 1)
        Y_hat = torch.mul(yhat, dag.unsqueeze(-1).unsqueeze(-1))
        Y = torch.mul(data.repeat(sz, 1).view(
            sz, sz, -1).unsqueeze(-1), dag.unsqueeze(-1).unsqueeze(-1))

        return Y, Y_hat, Z_mu, Z_logvar


def loss_function(y, yhat, mu, logvar, sdy, beta)->torch.Tensor:
    """Low level loss function(VAE).

    Args:
        y: raw data
        yhat: fitting data
        mu: VAE hidden output mu
        logvar:VAE hidden output logvar
        sdy: learnable parameter for sample distribution
        beta: amplification factor

    Returns:
        loss
    """
    N = y - yhat
    if sdy.item() <= 0:
        sdy = -sdy + 0.01

    n = torch.distributions.Normal(0, sdy)
    # Compute the log-likelihood of noise distribution.
    BCE = - torch.sum(n.log_prob(N), dim=(-1, -2))
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) -
                           logvar.exp(), dim=(-1, -2)) * beta
    return BCE + KLD
