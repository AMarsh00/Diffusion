"""
Displays expected primitive updates in the simple Gaussian case.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 2D Standard Gaussian
# ============================================================
def gaussian_2d(x):
    x = np.asarray(x)
    return (1.0 / (2.0 * np.pi)) * np.exp(-0.5 * np.sum(x**2, axis=-1))


def log_gaussian_2d(x):
    x = np.asarray(x)
    return -0.5 * np.sum(x**2, axis=-1) - np.log(2.0 * np.pi)


def score(x):
    """Exact score grad log p(x) for N(0, I_2)."""
    return -np.asarray(x)


def score_jvp(x, v):
    """Ds(x)[v] for s(x)=-x.  Here Ds=-I exactly."""
    del x
    return -np.asarray(v)


# ============================================================
# Current-paper metric g = I + lambda s s^T
# ============================================================
def g_metric(x, metric_lambda=1.0):
    s = score(x)
    return np.eye(2) + metric_lambda * np.outer(s, s)


def metric_apply(x, v, metric_lambda=1.0):
    """Apply g(x) to a tangent vector without forming unnecessary inverses."""
    s = score(x)
    return v + metric_lambda * s * np.dot(s, v)


def metric_norm_sq(x, v, metric_lambda=1.0):
    """||v||_{g(x)}^2 for the rank-one metric."""
    s = score(x)
    return float(np.dot(v, v) + metric_lambda * np.dot(s, v) ** 2)


# ============================================================
# Levi-Civita geodesic ODE
# ============================================================
def geodesic_acceleration(x, v, metric_lambda=1.0):
    r"""
    Current-paper Levi-Civita acceleration:

        x_ddot = -lambda/(1 + lambda ||s||^2)
                    s(x) (v^T Ds(x) v).

    For the standard Gaussian, s(x)=-x and Ds=-I, so the symmetry assumption
    is exact rather than approximate.
    """
    s = score(x)
    jv = score_jvp(x, v)
    inner = float(np.dot(v, jv))
    denom = 1.0 + metric_lambda * float(np.dot(s, s))
    return -(metric_lambda / denom) * s * inner


def riemannian_exponential(x0, v0, n_steps=32, metric_lambda=1.0):
    """Numerically integrate Exp_{x0}(v0) over unit geodesic time."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    x = np.asarray(x0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    dt = 1.0 / float(n_steps)

    # Same semi-implicit Euler update used by the image-space implementation.
    for _ in range(n_steps):
        a = geodesic_acceleration(x, v, metric_lambda=metric_lambda)
        v = v + dt * a
        x = x + dt * v

    return x


# ============================================================
# Riemannian logarithm via robust coarse-to-fine shooting
# ============================================================
def riemannian_logarithm(
    x,
    y,
    metric_lambda=1.0,
    substeps_schedule=(2, 4, 8, 16, 32, 64),
    total_iters=120,
    shooting_lr=0.5,
    tol=1e-8,
    backtracking_factor=0.5,
    max_backtracking_steps=10,
    return_error=False,
):
    r"""
    Approximate Log_x(y) by endpoint shooting.

    At each integration resolution S we iterate the paper-style residual update

        v <- v + eta (y - Exp_x(v)),

    but safeguard it with backtracking and carry the best tangent to the next
    (finer) integration resolution.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if metric_lambda == 0.0:
        v = y - x
        return (v, 0.0) if return_error else v

    v = (y - x).copy()
    n_stages = len(substeps_schedule)

    # Put more work into the finer stages.
    stage_weights = np.arange(1, n_stages + 1, dtype=float)
    stage_weights /= stage_weights.sum()
    stage_iters = np.maximum(3, np.floor(total_iters * stage_weights).astype(int))

    for stage_idx, n_steps in enumerate(substeps_schedule):
        best_v = v.copy()
        best_err = np.linalg.norm(
            y - riemannian_exponential(
                x, v, n_steps=n_steps, metric_lambda=metric_lambda
            )
        )
        lr_stage = float(shooting_lr)

        for _ in range(int(stage_iters[stage_idx])):
            exp_v = riemannian_exponential(
                x, v, n_steps=n_steps, metric_lambda=metric_lambda
            )
            residual = y - exp_v
            err = float(np.linalg.norm(residual))

            if err < best_err:
                best_err = err
                best_v = v.copy()

            if err <= tol:
                best_v = v.copy()
                best_err = err
                break

            accepted = False
            trial_lr = lr_stage

            for _ in range(max_backtracking_steps):
                v_trial = v + trial_lr * residual
                exp_trial = riemannian_exponential(
                    x,
                    v_trial,
                    n_steps=n_steps,
                    metric_lambda=metric_lambda,
                )
                trial_err = float(np.linalg.norm(y - exp_trial))

                if np.isfinite(trial_err) and trial_err < err:
                    v = v_trial
                    accepted = True
                    lr_stage = min(1.0, trial_lr * 1.15)
                    if trial_err < best_err:
                        best_err = trial_err
                        best_v = v_trial.copy()
                    break

                trial_lr *= backtracking_factor

            if not accepted:
                # Keep the best tangent seen at this resolution.  If even a very
                # small residual correction does not improve the endpoint, move
                # on to the next integration resolution.
                v = best_v.copy()
                break

        v = best_v.copy()

    final_steps = int(substeps_schedule[-1])
    final_pred = riemannian_exponential(
        x, v, n_steps=final_steps, metric_lambda=metric_lambda
    )
    final_err = float(np.linalg.norm(y - final_pred))

    return (v, final_err) if return_error else v


# ============================================================
# Localized Monte Carlo measure from the current paper
# ============================================================
def sample_local_proposal(rng, center, epsilon, N):
    """Sample Z_i ~ N(center, epsilon^2 I)."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if N < 1:
        raise ValueError("N must be positive")
    center = np.asarray(center, dtype=float)
    return center[None, :] + epsilon * rng.normal(size=(N, 2))


def normalized_importance_weights(samples):
    r"""
    For q=N(z_c,eps^2 I) and target nu proportional to K_eps d mu,

        w_i proportional to mu(Z_i) proportional to exp(-||Z_i||^2/2).
    """
    log_w = -0.5 * np.sum(samples**2, axis=1)
    log_w = log_w - np.max(log_w)
    w = np.exp(log_w)
    return w / np.sum(w)


# ============================================================
# Empirical localized Frechet objective
# ============================================================
def local_logs(
    z,
    samples,
    metric_lambda,
    shooting_schedule=(2, 4, 8, 16, 32),
    shooting_iters=60,
    shooting_lr=0.5,
):
    logs = []
    errors = []

    for zi in samples:
        L, err = riemannian_logarithm(
            z,
            zi,
            metric_lambda=metric_lambda,
            substeps_schedule=shooting_schedule,
            total_iters=shooting_iters,
            shooting_lr=shooting_lr,
            tol=1e-8,
            return_error=True,
        )
        logs.append(L)
        errors.append(err)

    return np.asarray(logs), np.asarray(errors)


def frechet_energy(
    z,
    samples,
    weights,
    metric_lambda,
    shooting_schedule=(2, 4, 8, 16, 32),
    shooting_iters=60,
    shooting_lr=0.5,
):
    logs, errors = local_logs(
        z,
        samples,
        metric_lambda,
        shooting_schedule=shooting_schedule,
        shooting_iters=shooting_iters,
        shooting_lr=shooting_lr,
    )

    d2 = np.array([
        metric_norm_sq(z, L, metric_lambda=metric_lambda)
        for L in logs
    ])
    energy = float(np.sum(weights * d2))
    return energy, logs, errors


# ============================================================
# Generator-constrained expected primitive / Frechet mean
# ============================================================
def expected_primitive(
    samples,
    weights,
    z_init,
    metric_lambda=1.0,
    step_size=0.20,
    Q=20,
    backtracking_factor=0.5,
    max_backtracking_steps=8,
    shooting_schedule=(2, 4, 8, 16, 32),
    shooting_iters=60,
    shooting_lr=0.5,
    grad_tol=1e-7,
):
    r"""
    Minimize the fixed empirical localized Frechet objective

        F(z) = sum_i w_i d_g(z, Z_i)^2.

    Here Phi=Id.  If

        Delta(z) = sum_i w_i Log_z(Z_i),

    then

        grad_g F = -2 Delta,
        grad_Euclidean F = -2 g(z) Delta.

    Hence d(z)=g(z)Delta is a Euclidean descent direction (factor 2 absorbed
    into the step size).  Backtracking is performed on the SAME fixed samples
    and normalized weights.
    """
    z = np.asarray(z_init, dtype=float).copy()
    trajectory = [z.copy()]
    energy_history = []
    shooting_error_history = []

    for q in range(Q):
        energy, logs, errors = frechet_energy(
            z,
            samples,
            weights,
            metric_lambda,
            shooting_schedule=shooting_schedule,
            shooting_iters=shooting_iters,
            shooting_lr=shooting_lr,
        )

        delta = np.sum(weights[:, None] * logs, axis=0)
        direction = metric_apply(z, delta, metric_lambda=metric_lambda)
        direction_norm = float(np.linalg.norm(direction))

        energy_history.append(energy)
        shooting_error_history.append(errors.copy())

        print(
            f"Iter {q + 1:02d}/{Q}: "
            f"F={energy:.8f}, |d|={direction_norm:.6g}, "
            f"log err mean={errors.mean():.3e}, max={errors.max():.3e}"
        )

        if direction_norm <= grad_tol:
            print("  descent direction below tolerance; stopping")
            break

        accepted = False
        alpha = float(step_size)

        for _ in range(max_backtracking_steps):
            z_trial = z + alpha * direction
            trial_energy, _, _ = frechet_energy(
                z_trial,
                samples,
                weights,
                metric_lambda,
                shooting_schedule=shooting_schedule,
                shooting_iters=shooting_iters,
                shooting_lr=shooting_lr,
            )

            if np.isfinite(trial_energy) and trial_energy <= energy:
                z = z_trial
                trajectory.append(z.copy())
                print(
                    f"  accepted step={alpha:.6g}: "
                    f"F {energy:.8f} -> {trial_energy:.8f}"
                )
                accepted = True
                break

            alpha *= backtracking_factor

        if not accepted:
            print("  no decreasing trial step found; stopping")
            break

    return (
        np.asarray(trajectory),
        np.asarray(energy_history),
        shooting_error_history,
    )


# ============================================================
# Run experiment
# ============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(5)

    # --------------------------------------------------------
    # Localized measure parameters
    # --------------------------------------------------------
    localization_center = np.array([1.35, 0.85])
    epsilon = 0.75
    N = 32

    # Current-paper metric parameter lambda.
    metric_lambda = 0.50

    # Draw proposal samples and construct fixed normalized importance weights.
    samples = sample_local_proposal(
        rng,
        center=localization_center,
        epsilon=epsilon,
        N=N,
    )
    weights = normalized_importance_weights(samples)

    # Optimization initialization, kept separate from the localization center.
    # We choose a point far enough away to make the descent trajectory visible,
    # but still within a regime where the local shooting solver is well conditioned.
    z_init = np.array([-0.8, 1.2])

    print("Localized standard-Gaussian Frechet-mean experiment")
    print(f"  localization center = {localization_center}")
    print(f"  epsilon             = {epsilon}")
    print(f"  N                   = {N}")
    print(f"  lambda              = {metric_lambda}")
    print(f"  initial point       = {z_init}")
    print(f"  importance ESS      = {1.0 / np.sum(weights**2):.3f} / {N}")

    traj, energy_history, shooting_errors = expected_primitive(
        samples,
        weights,
        z_init,
        metric_lambda=metric_lambda,
        step_size=0.20,
        Q=18,
        backtracking_factor=0.5,
        max_backtracking_steps=8,
        shooting_schedule=(2, 4, 8, 16, 32),
        shooting_iters=50,
        shooting_lr=0.5,
        grad_tol=1e-7,
    )

    # Exact Euclidean mean of the localized target measure.  This is NOT the
    # Riemannian Frechet mean for lambda>0; it is shown only as a useful sanity
    # reference for the Gaussian localization itself.
    exact_localized_euclidean_mean = localization_center / (1.0 + epsilon**2)

    final_energy, _, final_errors = frechet_energy(
        traj[-1],
        samples,
        weights,
        metric_lambda,
        shooting_schedule=(2, 4, 8, 16, 32),
        shooting_iters=50,
        shooting_lr=0.5,
    )

    print("\nFinal diagnostics")
    print(f"  final point                    = {traj[-1]}")
    print(f"  final empirical F              = {final_energy:.8f}")
    print(
        f"  final log error mean/max       = "
        f"{final_errors.mean():.3e} / {final_errors.max():.3e}"
    )
    print(
        "  exact localized Euclidean mean = "
        f"{exact_localized_euclidean_mean}  (reference only)"
    )

    # ========================================================
    # Plot
    # ========================================================
    x = np.linspace(-3.0, 3.0, 250)
    y = np.linspace(-3.0, 3.0, 250)
    X, Y = np.meshgrid(x, y)
    Z = gaussian_2d(np.stack([X, Y], axis=-1))

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=40, cmap="viridis")

    # Proposal samples; marker sizes visualize their normalized importance weights.
    sizes = 30.0 + 260.0 * weights / weights.max()
    plt.scatter(
        samples[:, 0],
        samples[:, 1],
        s=sizes,
        alpha=0.55,
        label="Local proposal samples",
    )

    plt.plot(
        traj[:, 0],
        traj[:, 1],
        "-o",
        lw=2,
        label="Localized Frechet-mean updates",
    )

    plt.scatter(
        z_init[0],
        z_init[1],
        s=90,
        marker="s",
        label="Optimization start",
    )
    plt.scatter(
        traj[-1, 0],
        traj[-1, 1],
        s=100,
        marker="*",
        label="Estimated Frechet mean",
    )
    plt.scatter(
        localization_center[0],
        localization_center[1],
        s=100,
        marker="x",
        label="Localization center",
    )
    plt.scatter(
        exact_localized_euclidean_mean[0],
        exact_localized_euclidean_mean[1],
        s=90,
        marker="D",
        label="Exact localized Euclidean mean",
    )

    plt.legend()
    plt.title("Localized Frechet Mean for a 2D Standard Gaussian")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Gaussian density")
    plt.tight_layout()
    plt.savefig("gaussian_expected_primitive_updated.png", dpi=180, bbox_inches="tight")
    plt.show()
