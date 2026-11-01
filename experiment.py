"""Paired SΩI experiments: train once per case, compare two formulas."""
import argparse
import copy
import csv
import itertools
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np
from scipy.special import digamma, gammaln, logsumexp
from scipy.spatial import cKDTree
import torch

from model import UnetMLP_simple

SEED = 202
STEPS = 30_000
N = 6_000
VARIABLES = 4
BLOCK_DIM = 3
CONFIGS = [
    dict(family='gaussian', strength=.5, structure='red', transform='halfcube'),
    dict(family='gaussian', strength=.7, structure='syn', transform='identity'),
    *[dict(family='common_mixture', strength=a) for a in [.8, 2.]],
    *[dict(family='parity_mixture', strength=a) for a in [1., 2.]],
    *[dict(family='scale_mixture', strength=a) for a in [.1, .4]],
    *[dict(family='student', strength=nu) for nu in [3., 8.]],
]
CSV_FIELDS = ['family', 'strength', 'structure', 'transform', 'seed',
              'reference_omega', 'reference_mc_se', 'original', 'proposed', 'ksg',
              'original_mc_se', 'proposed_mc_se']


def covariance(cfg):
    rho = cfg['strength'] if cfg['family'] == 'gaussian' else .35
    if cfg.get('structure', 'red') == 'red':
        return (1-rho)*np.eye(VARIABLES)+rho*np.ones((VARIABLES, VARIABLES))
    cov = np.eye(VARIABLES)
    cov[0, 1] = cov[1, 0] = 1/np.sqrt(VARIABLES-1)
    for i in range(2, VARIABLES):
        cov[1, i] = cov[i, 1] = rho/np.sqrt(VARIABLES-1)
    return cov


def mixture(cfg):
    family, a = cfg['family'], cfg['strength']
    if family == 'common_mixture':
        return np.array([-np.ones(VARIABLES)*a, np.ones(VARIABLES)*a]), np.ones(2), np.array([.5, .5])
    if family == 'parity_mixture':
        signs = np.array([s for s in itertools.product([-1, 1], repeat=VARIABLES)
                          if np.prod(s) == 1])
        return a*signs, np.ones(len(signs)), np.ones(len(signs))/len(signs)
    return np.zeros((2, VARIABLES)), np.array([a, (1-.8*a)/.2]), np.array([.8, .2])


def sample_channel(cfg, n, seed):
    rng = np.random.default_rng(seed)
    if cfg['family'] in ['gaussian', 'student']:
        x = rng.normal(size=(n, VARIABLES)) @ np.linalg.cholesky(covariance(cfg)).T
        if cfg['family'] == 'student':
            nu = cfg['strength']
            return x/np.sqrt(rng.chisquare(nu, size=(n, 1))/nu)
        return x*np.sqrt(np.abs(x)) if cfg.get('transform') == 'halfcube' else x
    means, variances, weights = mixture(cfg)
    labels = rng.choice(len(weights), n, p=weights)
    return means[labels] + np.sqrt(variances[labels, None])*rng.normal(size=(n, VARIABLES))


def sample(cfg, n, seed):
    channels = [sample_channel(cfg, n, seed + 100003*j) for j in range(BLOCK_DIM)]
    return np.stack(channels, axis=2).reshape(n, VARIABLES*BLOCK_DIM)


def reference_omega(cfg):
    """Analytic or independent known-density reference; never used in training."""
    if cfg['family'] in ['gaussian', 'student']:
        cov = covariance(cfg)
        def entropy(c):
            d = len(c)
            if cfg['family'] == 'gaussian':
                return .5*np.linalg.slogdet(c)[1]
            nu = cfg['strength']
            return (gammaln(nu/2)-gammaln((nu+d)/2)
                    + d/2*np.log(nu*np.pi)+.5*np.linalg.slogdet(c)[1]
                    + (nu+d)/2*(digamma((nu+d)/2)-digamma(nu/2)))
        value = (VARIABLES-2)*entropy(cov)
        for i in range(VARIABLES):
            keep = np.arange(VARIABLES) != i
            value += entropy(cov[i:i+1, i:i+1])
            value -= entropy(cov[np.ix_(keep, keep)])
        return float(BLOCK_DIM*value), 0.

    means, variances, weights = mixture(cfg)
    def logpdf(x, keep):
        x, centers = x[:, keep], means[:, keep]
        d = x.shape[1]
        squared = ((x[:, None, :] - centers[None, :, :])**2).sum(-1)
        components = (np.log(weights)[None, :]
                      - d/2*np.log(2*np.pi*variances)[None, :]
                      - squared/(2*variances[None, :]))
        return logsumexp(components, axis=1)

    values = []
    for j in range(40):
        x = sample_channel(cfg, 10_000, 987123 + j*391)
        value = -(VARIABLES-2)*logpdf(x, np.arange(VARIABLES))
        for i in range(VARIABLES):
            value -= logpdf(x, [i])
            value += logpdf(x, np.arange(VARIABLES) != i)
        values.append(value)
    values = np.concatenate(values)
    # Independent channels make Ω additive. All channels have the same law.
    return (float(BLOCK_DIM*values.mean()),
            float(BLOCK_DIM*values.std(ddof=1)/np.sqrt(len(values))))


def ksg_omega(x, k=5):
    """KSG-1 with common joint radii, on all N observations."""
    x = (x-x.mean(0))/np.maximum(x.std(0), 1e-10)
    n = len(x)
    radii = cKDTree(x).query(x, k=k+1, p=np.inf, workers=1)[0][:, -1]
    if np.any(radii <= 0):
        raise ValueError('Tied observations')
    radii = np.nextafter(radii, 0)
    singles, complements = np.zeros(n), np.zeros(n)
    for i in range(VARIABLES):
        keep = np.arange(VARIABLES*BLOCK_DIM)//BLOCK_DIM == i
        for columns, counts in [(keep, singles), (~keep, complements)]:
            subset = x[:, columns]
            # Counts include the observation itself.
            counts += digamma(cKDTree(subset).query_ball_point(
                subset, radii, p=np.inf, return_length=True, workers=1))
    return float((2-VARIABLES)*digamma(k)+(VARIABLES-2)*digamma(n)
                 + np.mean(complements-singles))


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


def train(x, seed=SEED):
    torch.manual_seed(seed + 10000)
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
def estimate(net, center, scale, x, seed=SEED):
    data = torch.tensor((x-center)/scale, dtype=torch.float32)
    rng = torch.Generator().manual_seed(seed + 20000)
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


def run(cfg, seed, reference=None):
    started = time.perf_counter()
    data = sample(cfg, N, seed)
    print(f"{cfg['family']} {cfg['strength']}, seed {seed}", flush=True)
    net, center, scale = train(data[:4200], seed)
    estimates = estimate(net, center, scale, data[5100:], seed)
    estimates['ksg'] = {'omega': ksg_omega(data)}
    reference, reference_se = reference or reference_omega(cfg)
    for value in estimates.values():
        value['absolute_error'] = abs(value['omega']-reference)
    reduction = 1-estimates['proposed']['absolute_error']/estimates['original']['absolute_error']
    result = {'config': cfg, 'seed': seed, 'training_steps': STEPS, 'observations': N,
              'train_n': 4200, 'evaluation_n': 900, 'variables': VARIABLES,
              'coordinates_per_variable': BLOCK_DIM,
              'reference_omega': reference, 'reference_mc_se': reference_se,
              'estimates': estimates, 'relative_absolute_error_reduction': reduction,
              'seconds': time.perf_counter()-started}
    print(f'Reference Ω: {reference:.6f} (MC SE {reference_se:.6f})')
    print(f'{"Formula":12s} {"Omega":>12s} {"Abs. error":>12s}')
    for name, value in estimates.items():
        print(f'{name:12s} {value["omega"]:12.6f} {value["absolute_error"]:12.6f}')
    print(f'Absolute error reduction: {100*reduction:.2f}%')
    print(f'Elapsed: {result["seconds"]:.1f}s.', flush=True)
    return result


def csv_row(result):
    cfg, estimates = result['config'], result['estimates']
    return {**{k: cfg.get(k, 'identity' if k == 'transform' else '')
               for k in ['family', 'strength', 'structure', 'transform']},
            **{k: result[k] for k in ['seed', 'reference_omega', 'reference_mc_se']},
            **{k: estimates[k]['omega'] for k in ['original', 'proposed', 'ksg']},
            **{k+'_mc_se': estimates[k]['conditional_mc_se'] for k in ['original', 'proposed']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Run all 20 confirmation cases')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not args.all:
        result = run(CONFIGS[4], SEED)
        Path('result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print('Saved result.json.')
        return

    rows, references = [], {}
    with Path('run_results.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for seed in [202, 303]:
            for i, cfg in enumerate(CONFIGS):
                if i not in references:
                    references[i] = reference_omega(cfg)
                row = csv_row(run(cfg, seed, references[i]))
                writer.writerow(row)
                output.flush()
                rows.append(row)
    errors = {name: np.abs([r[name]-r['reference_omega'] for r in rows])
              for name in ['original', 'proposed', 'ksg']}
    for name, values in errors.items():
        print(f'{name:12s} MAE: {values.mean():.6f}')
    print(f"New formula wins: {np.sum(errors['proposed'] < errors['original'])}/{len(rows)}")
    print('Saved run_results.csv.')


if __name__ == '__main__':
    main()
