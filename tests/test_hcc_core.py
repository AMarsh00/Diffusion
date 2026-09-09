import sys
from pathlib import Path

import torch

# Run with: pytest -q
ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "Code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import High_Confidence_Curves as hcc
from experiment_utils import strict_load_model


class ZeroModel(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_direct_localized_sampler_matches_analytic_mean_and_variance():
    torch.manual_seed(0)
    centers = torch.tensor([[1.0, -0.5, 0.25]])
    sigma = 0.4
    n = 20000
    z = hcc.sample_localized_measure_direct(centers, n, sigma, antithetic=False)[0]
    denom = 1.0 + sigma ** 2
    target_mean = centers[0] / denom
    target_var = sigma ** 2 / denom
    assert torch.allclose(z.mean(dim=0), target_mean, atol=0.015, rtol=0.0)
    assert torch.allclose(z.var(dim=0, unbiased=True), torch.full_like(target_mean, target_var), atol=0.015, rtol=0.0)


def test_importance_weights_are_normalized_and_positive():
    torch.manual_seed(0)
    centers = torch.randn(3, 1, 2, 2)
    _, w = hcc.sample_local_proposal_with_importance_weights(centers, 8, 0.2)
    assert torch.all(w >= 0)
    assert torch.allclose(w.sum(dim=1), torch.ones(3), atol=1e-6)


def test_hcc_preserves_endpoints_and_accepted_energy_is_monotone_per_step():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = ZeroModel().to(device)
    scheduler = hcc.VPScheduler(num_timesteps=10)

    curve = torch.randn(5, 1, 2, 2)
    endpoint_a = curve[0].clone()
    endpoint_b = curve[-1].clone()

    def phi(z):
        return z

    def phi_grad(z):
        return z

    def exp_map(x, v):
        return x + v

    history, pre, accepted, _ = hcc.refine_latent_constrained_clean_frechet(
        curve,
        latent_to_clean_fn=phi,
        latent_to_clean_grad_fn=phi_grad,
        exp_map_fn=exp_map,
        model=model,
        scheduler=scheduler,
        proxy_t=5,
        metric_lambda=0.0,
        sigma=0.2,
        n_iters=2,
        n_candidates=4,
        latent_step_size=0.05,
        step_decay=1.0,
        min_latent_step=0.05,
        local_shooting_iters=1,
        local_shooting_lr=1.0,
        pair_chunk_size=16,
        normalize_step=True,
        backtracking_factor=0.5,
        max_backtracking_steps=2,
        energy_decrease_tol=0.0,
    )
    assert torch.equal(history[-1][0], endpoint_a)
    assert torch.equal(history[-1][-1], endpoint_b)
    for before, after in zip(pre, accepted):
        assert torch.all(after <= before + 1e-6)


def test_missing_checkpoint_fails_loudly(tmp_path):
    missing = tmp_path / "does_not_exist.pt"
    try:
        strict_load_model(missing, torch.device("cpu"))
    except FileNotFoundError as exc:
        assert "Refusing to run with random weights" in str(exc)
    else:
        raise AssertionError("Missing checkpoint must raise FileNotFoundError")
