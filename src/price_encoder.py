"""EMA filter + single-layer RNN price feature extractor (paper Sec. III-A, Eq. 6-8).

    rho_tilde[t+1] = alpha * rho_tilde[t] + (1 - alpha) * rho[t+1]      (Eq. 6, EMA smoothing)
    h[t+1]         = tanh(W @ h[t] + w @ rho_tilde[t+1] + b)            (Eq. 7, RNN update)
    rho_hat[t+1]   = w_o @ h[t] + b_o                                   (Eq. 8, next-price prediction)

Trained via backprop-through-time on its OWN auxiliary loss -- predicting
the next EMA-smoothed price, minimizing sum (rho_hat[t+1] - rho_tilde[t+1])^2
-- entirely separately from the RL reward. Eq. 8 predicts rho_tilde[t+1]
using h[t] (the hidden state from *before* rho[t+1] was observed), which is
what makes this a genuine forecast rather than circular -- implemented
below by feeding the output head h shifted by one step.

Because h[t] depends only on the price sequence up to t -- never on the
agent's actions -- it can be computed once for an entire price series after
training and then just looked up during RL rollouts, rather than run live
inside the environment. This project trains the encoder once (pretraining,
not joint end-to-end training with the PPO objective): the paper presents
Sec. III-A's auxiliary loss independently of Sec. III-B's RL algorithm, and
doesn't state whether the two are trained jointly or the encoder is frozen
before RL training -- pretrain-then-freeze is the simpler, and more
directly supported, reading, but it's a real assumption where the paper
doesn't fully specify the procedure.

Uses nn.RNN (not a manual per-timestep RNNCell Python loop) for the actual
recurrence -- mathematically identical, but running BPTT for 4000 steps
over an ~6,500-hour training sequence in a pure Python per-timestep loop
would be prohibitively slow. The EMA itself has no learnable part (alpha is
a fixed hyperparameter), so it's computed once up front rather than
recomputed on every one of those 4000 forward passes.
"""

import numpy as np
import torch
import torch.nn as nn


def compute_ema(prices, alpha):
    """rho_tilde (Eq. 6), as a plain numpy array -- no gradient needed since
    alpha is fixed and this doesn't depend on any learnable parameter.
    """
    prices = np.asarray(prices, dtype=np.float32)
    rho_tilde = np.empty_like(prices)
    rho_tilde[0] = prices[0]
    for t in range(1, len(prices)):
        rho_tilde[t] = alpha * rho_tilde[t - 1] + (1 - alpha) * prices[t]
    return rho_tilde


class PriceEncoder(nn.Module):
    def __init__(self, hidden_size=16, alpha=0.7):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, num_layers=1,
                           nonlinearity="tanh", batch_first=True)
        self.output_head = nn.Linear(hidden_size, 1)  # w_o, b_o (Eq. 8)

    def forward(self, rho_tilde):
        """rho_tilde: (T,) tensor, precomputed via compute_ema.

        Returns:
          hiddens: (T, hidden_size) -- hiddens[t] = h_{t+1} in the paper's
                   1-indexed notation (built from rho_tilde[0..t] inclusive).
          preds:   (T,) -- preds[t] = rho_hat_{t+1}, predicted from h_t
                   (the hidden state *before* incorporating rho_tilde[t]),
                   i.e. a genuine one-step-ahead forecast comparable against
                   rho_tilde[t] in the auxiliary loss.
        """
        rnn_input = rho_tilde.view(1, -1, 1)  # (batch=1, seq=T, input_size=1)
        hiddens, _ = self.rnn(rnn_input)  # h0 defaults to zeros; hiddens: (1, T, hidden_size)
        hiddens = hiddens.squeeze(0)

        h0 = torch.zeros(1, self.hidden_size)
        prev_hiddens = torch.cat([h0, hiddens[:-1]], dim=0)  # prev_hiddens[t] = h_t
        preds = self.output_head(prev_hiddens).squeeze(-1)
        return hiddens, preds


def fit_price_encoder(prices, hidden_size=16, alpha=0.7, n_steps=4000, lr=0.01, seed=0):
    """Train on one price series via the auxiliary loss (paper: 4000 steps,
    lr=0.01, Adam). Full-sequence BPTT each step -- an hourly year is only a
    few thousand steps, small enough to do directly without truncation.
    """
    torch.manual_seed(seed)
    encoder = PriceEncoder(hidden_size=hidden_size, alpha=alpha)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    rho_tilde = torch.as_tensor(compute_ema(prices, alpha))

    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        _, preds = encoder(rho_tilde)
        loss = torch.mean((preds - rho_tilde) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 500 == 0:
            print(f"  price encoder step {step + 1}/{n_steps}: loss={loss.item():.4f}")

    return encoder, losses


def compute_hidden_states(encoder, prices):
    """Frozen-encoder feature extraction: (T, hidden_size) numpy array."""
    encoder.eval()
    with torch.no_grad():
        rho_tilde = torch.as_tensor(compute_ema(prices, encoder.alpha))
        hiddens, _ = encoder(rho_tilde)
    return hiddens.numpy()
