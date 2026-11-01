"""One paired SΩI experiment: train once, compare two integration formulas."""
import copy
import itertools
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np
from scipy.special import logsumexp
import torch

from model import UnetMLP_simple

SEED = 202
STEPS = 30_000
N = 6_000
VARIABLES = 4
BLOCK_DIM = 3
MEANS = np.array([s for s in itertools.product([-1, 1], repeat=VARIABLES)
                  if np.prod(s) == 1], dtype=float)
MIXTURE_WEIGHTS = np.ones(len(MEANS)) / len(MEANS)


def sample_channel(n, seed):
    rng = np.random.default_rng(seed)
    labels = rng.choice(len(MEANS), n, p=MIXTURE_WEIGHTS)
    return MEANS[labels] + rng.normal(size=(n, VARIABLES))


def sample(n, seed):
    channels = [sample_channel(n, seed + 100003*j) for j in range(BLOCK_DIM)]
    return np.stack(channels, axis=2).reshape(n, VARIABLES*BLOCK_DIM)


def reference_omega():
    """Independent known-density reference, never used to train the network."""
    def logpdf(x, keep):
        x, means = x[:, keep], MEANS[:, keep]
        d = x.shape[1]
        squared = ((x[:, None, :] - means[None, :, :])**2).sum(-1)
        components = (np.log(MIXTURE_WEIGHTS)[None, :]
                      - d/2*np.log(2*np.pi) - squared/2)
        return logsumexp(components, axis=1)

    values = []
    for j in range(40):
        x = sample_channel(10_000, 987123 + j*391)
        value = -(VARIABLES-2)*logpdf(x, np.arange(VARIABLES))
        for i in range(VARIABLES):
            value -= logpdf(x, [i])
            value += logpdf(x, np.arange(VARIABLES) != i)
        values.append(value)
    values = np.concatenate(values)
    # Independent channels make Ω additive. All channels have the same law.
    return (float(BLOCK_DIM*values.mean()),
            float(BLOCK_DIM*values.std(ddof=1)/np.sqrt(len(values))))


def vp_parameters(lam):
    alpha, sigma = torch.sigmoid(lam).sqrt(), torch.sigmoid(-lam).sqrt()
    t = (-.1 + torch.sqrt(.01 + 39.8*torch.nn.functional.softplus(-lam)))/19.9
    return alpha, sigma, t


def masks(training=False):
    joint = np.ones((1, VARIABLES))
    marginal = 2*np.eye(VARIABLES)-1
    conditional = np.eye(VARIABLES)
    if training:
        joint = np.repeat(joint, VARIABLES, axis=0)
        conditional = np.repeat(conditional, 1+(VARIABLES-1)//2, axis=0)
    return torch.tensor(np.concatenate([joint, marginal, conditional]), dtype=torch.float32)


def predict(net, clean, noisy, t, mask, filler):
    expanded = mask.repeat_interleave(BLOCK_DIM, dim=-1)
    inputs = torch.where(expanded > 0, noisy, torch.where(expanded == 0, clean, filler))
    times = t*mask.clamp(0, 1) + (mask < 0).float()
    return net(inputs, t=times)


def train(x):
    torch.manual_seed(SEED + 10000)
    center, scale = x.mean(0), np.maximum(x.std(0), 1e-8)
    data = torch.tensor((x-center)/scale, dtype=torch.float32)
    net = UnetMLP_simple(dim=VARIABLES*BLOCK_DIM, init_dim=128,
                        dim_mults=[], time_dim=128, nb_var=VARIABLES)
    ema = copy.deepcopy(net).eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    bank = masks(training=True)
    for step in range(STEPS):
        batch = data[torch.randint(len(data), (256,))]
        mask = bank[torch.randint(len(bank), (256,))]
        lam = -10 + 20*torch.rand(256, 1)
        alpha, sigma, t = vp_parameters(lam)
        noise = torch.randn_like(batch)
        noisy = alpha*batch + sigma*noise
        prediction = predict(net, batch, noisy, t, mask, torch.randn_like(batch))
        active = (mask > 0).float().repeat_interleave(BLOCK_DIM, dim=1)
        weight = 2-active.sum(1)/(VARIABLES*BLOCK_DIM)
        loss = (((prediction-noise)**2)*active).sum(1).mul(weight).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            decay = min(.999, (step+1)/(step+10))
            for averaged, parameter in zip(ema.parameters(), net.parameters()):
                averaged.mul_(decay).add_(parameter, alpha=1-decay)
        if (step+1) % 5000 == 0:
            print(f'Training: {step+1}/{STEPS}', flush=True)
    return ema, center, scale


@torch.no_grad()
def estimate(net, center, scale, x):
    data = torch.tensor((x-center)/scale, dtype=torch.float32)
    rng = torch.Generator().manual_seed(SEED + 20000)
    noise = torch.randn(data.shape, generator=rng)
    filler = torch.randn(data.shape, generator=rng)
    points, weights = np.polynomial.legendre.leggauss(32)
    points, weights = points*10, weights*5  # 1/2 integral over log-SNR [-10,10].
    old, new = [], []
    for lam in points:
        alpha, sigma, t = vp_parameters(torch.full((len(data), 1), lam))
        noisy = alpha*data + sigma*noise
        predictions = [predict(net, data, noisy, t, mask, filler) for mask in masks()]
        joint = predictions[0].numpy()
        marginal = torch.cat([predictions[1+i][:, i*BLOCK_DIM:(i+1)*BLOCK_DIM]
                              for i in range(VARIABLES)], dim=1).numpy()
        conditional = torch.cat([predictions[1+VARIABLES+i][:, i*BLOCK_DIM:(i+1)*BLOCK_DIM]
                                 for i in range(VARIABLES)], dim=1).numpy()
        eps = noise.numpy()
        old.append(((joint-marginal)**2).sum(1) - ((joint-conditional)**2).sum(1))
        # ||eps-M||² + ||eps-C||² - 2||eps-J||², with eps² cancelled.
        new.append((marginal*marginal + conditional*conditional - 2*joint*joint
                    + 2*eps*(2*joint-marginal-conditional)).sum(1))
    result = {}
    for name, integrand in [('original', old), ('proposed', new)]:
        rows = weights @ np.asarray(integrand)
        result[name] = {'omega': float(rows.mean()),
                        'conditional_mc_se': float(rows.std(ddof=1)/np.sqrt(len(rows)))}
    return result


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    data = sample(N, SEED)
    print('One parity-mixture experiment; CPU training from scratch.', flush=True)
    net, center, scale = train(data[:4200])
    estimates = estimate(net, center, scale, data[5100:])
    reference, reference_se = reference_omega()
    for value in estimates.values():
        value['absolute_error'] = abs(value['omega']-reference)
    reduction = 1-estimates['proposed']['absolute_error']/estimates['original']['absolute_error']
    result = {'seed': SEED, 'training_steps': STEPS, 'observations': N,
              'train_n': 4200, 'evaluation_n': 900, 'variables': VARIABLES,
              'coordinates_per_variable': BLOCK_DIM,
              'reference_omega': reference, 'reference_mc_se': reference_se,
              'estimates': estimates, 'relative_absolute_error_reduction': reduction,
              'seconds': time.perf_counter()-started}
    Path('result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f'\nReference Ω: {reference:.6f} (MC SE {reference_se:.6f})')
    print(f'{"Formula":12s} {"Omega":>12s} {"Abs. error":>12s}')
    for name, value in estimates.items():
        print(f'{name:12s} {value["omega"]:12.6f} {value["absolute_error"]:12.6f}')
    print(f'Absolute error reduction: {100*reduction:.2f}%')
    print(f'Elapsed: {result["seconds"]:.1f}s. Saved result.json.')


if __name__ == '__main__':
    main()
