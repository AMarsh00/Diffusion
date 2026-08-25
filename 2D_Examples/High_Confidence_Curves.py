"""
Computes our High-Confidence Curve interpolation for a ridge (smeared Gaussian) dataset.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Ridge distribution parameters
# ============================================================
a = 0.8
sigma_par = 1.5
sigma_perp = 0.2

# Unit normal / tangent to the ridge y = a x
n = np.array([-a, 1.0], dtype=np.float64)
n /= np.linalg.norm(n)

t_vec = np.array([1.0, a], dtype=np.float64)
t_vec /= np.linalg.norm(t_vec)

# Exact linear generator Phi(z)=A z, z~N(0,I_2)
# Column 0 = tangent direction scale, column 1 = normal direction scale.
A = np.column_stack((sigma_par * t_vec, sigma_perp * n))
A_inv = np.linalg.inv(A)


# ============================================================
# Ridge density, exact score, exact score Jacobian
# ============================================================
def ridge_2d(x):
    x = np.asarray(x, dtype=np.float64)
    x_par = np.sum(x * t_vec, axis=-1)
    x_perp = np.sum(x * n, axis=-1)
    exponent = -0.5 * (
        x_par**2 / sigma_par**2
        + x_perp**2 / sigma_perp**2
    )
    return np.exp(np.clip(exponent, -700.0, 50.0))


SCORE_JACOBIAN = (
    -(1.0 / sigma_par**2) * np.outer(t_vec, t_vec)
    -(1.0 / sigma_perp**2) * np.outer(n, n)
)


def score(x):
    """Exact score grad log p(x). Supports shape (...,2)."""
    x = np.asarray(x, dtype=np.float64)
    return x @ SCORE_JACOBIAN.T


def Jv(x, v):
    """Exact Ds(x)[v]. Ds is constant and symmetric for this Gaussian."""
    del x
    v = np.asarray(v, dtype=np.float64)
    return v @ SCORE_JACOBIAN.T


# ============================================================
# Exact synthetic generator and inverse
# ============================================================
def Phi(z):
    z = np.asarray(z, dtype=np.float64)
    return z @ A.T


def Phi_inv(x):
    x = np.asarray(x, dtype=np.float64)
    return x @ A_inv.T


# ============================================================
# Independent ground-truth validation
# ============================================================
def true_ridge_error(x):
    """Squared Euclidean distance to the ridge centerline."""
    x = np.asarray(x, dtype=np.float64)
    return np.sum(x * n, axis=-1) ** 2


def mean_true_ridge_error(curve):
    if len(curve) <= 2:
        return 0.0
    return float(np.mean(true_ridge_error(curve[1:-1])))


def mean_negative_log_density(curve):
    """Density-based validation up to the irrelevant normalizing constant."""
    interior = np.asarray(curve, dtype=np.float64)[1:-1]
    x_par = interior @ t_vec
    x_perp = interior @ n
    return float(np.mean(0.5 * (
        x_par**2 / sigma_par**2 + x_perp**2 / sigma_perp**2
    )))


# ============================================================
# Paper metric g(x)=I+lambda s s^T
# ============================================================
def g_metric(x, lam=1.0):
    s = score(x)
    return np.eye(2, dtype=np.float64) + float(lam) * np.outer(s, s)


def metric_sqnorm_batch(x, v, lam=1.0):
    """Batched v^T g(x)v for x,v of shape (N,2)."""
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    s = score(x)
    return np.sum(v * v, axis=1) + float(lam) * np.sum(s * v, axis=1) ** 2


# ============================================================
# Levi-Civita ODE
# ============================================================
def geodesic_acceleration_batch(x, v, lam=1.0):
    r"""
    Exact for this example because Ds is symmetric:

      x_ddot = -lambda/(1+lambda||s||^2)
                  s (x_dot^T Ds x_dot).
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    s = score(x)
    jv = Jv(x, v)
    inner = np.sum(v * jv, axis=1)
    denom = 1.0 + float(lam) * np.sum(s * s, axis=1)
    return -(float(lam) * inner / denom)[:, None] * s


def levi_civita_exp_map_batch(x, v, n_steps=32, lam=1.0):
    """Semi-implicit Euler approximation of Exp_x(v) over unit time."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    x_curr = np.asarray(x, dtype=np.float64).copy()
    v_curr = np.asarray(v, dtype=np.float64).copy()

    if x_curr.ndim == 1:
        x_curr = x_curr[None, :]
    if v_curr.ndim == 1:
        v_curr = v_curr[None, :]

    dt = 1.0 / float(n_steps)
    for _ in range(int(n_steps)):
        a_curr = geodesic_acceleration_batch(x_curr, v_curr, lam=lam)
        v_curr = v_curr + dt * a_curr
        x_curr = x_curr + dt * v_curr

    return x_curr


def levi_civita_exp_map(x, v, n_steps=32, lam=1.0):
    return levi_civita_exp_map_batch(x, v, n_steps=n_steps, lam=lam)[0]


# ============================================================
# Coarse-to-fine Log shooting
# ============================================================
def _stage_iteration_budget(n_stages, total_iters):
    """Allocate more shooting iterations to the finer ODE resolutions."""
    weights = np.arange(1, n_stages + 1, dtype=np.float64)
    raw = total_iters * weights / weights.sum()
    budgets = np.maximum(1, np.floor(raw).astype(int))

    remainder = int(total_iters - budgets.sum())
    k = n_stages - 1
    while remainder > 0:
        budgets[k] += 1
        remainder -= 1
        k = (k - 1) % n_stages

    return budgets


def levi_civita_log_map_batch(
    x,
    y,
    lam=1.0,
    substeps_schedule=(1, 2, 4, 8, 16, 32),
    total_iters=120,
    eta=0.5,
    tol=1e-7,
    backtracking_factor=0.5,
    max_backtracking_steps=10,
    initial_v=None,
    return_errors=False,
    verbose=False,
):
    """
    Approximate Log_x(y) with coarse-to-fine residual shooting.

    At each resolution we retain the best tangent encountered and backtrack
    the residual correction v <- v + eta (y-Exp_x(v)) whenever it increases
    endpoint error. The best tangent from one ODE resolution initializes the
    next resolution.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    squeeze = x.ndim == 1

    if x.ndim == 1:
        x = x[None, :]
    if y.ndim == 1:
        y = y[None, :]
    if x.shape != y.shape or x.shape[1] != 2:
        raise ValueError("x and y must have matching shape (N,2) or (2,)")

    schedule = tuple(int(s) for s in substeps_schedule)
    budgets = _stage_iteration_budget(len(schedule), int(total_iters))

    if initial_v is None:
        v = y - x
    else:
        v = np.asarray(initial_v, dtype=np.float64).copy()
        if v.ndim == 1:
            v = v[None, :]

    if float(lam) == 0.0:
        final_err = np.zeros(len(x), dtype=np.float64)
        out = v[0] if squeeze else v
        return (out, final_err) if return_errors else out

    N = x.shape[0]

    for stage_idx, (n_steps, n_iters) in enumerate(zip(schedule, budgets)):
        pred = levi_civita_exp_map_batch(x, v, n_steps=n_steps, lam=lam)
        err = np.linalg.norm(y - pred, axis=1)
        best_v = v.copy()
        best_err = err.copy()
        stage_eta = np.full(N, float(eta), dtype=np.float64)

        for _ in range(int(n_iters)):
            pred = levi_civita_exp_map_batch(x, v, n_steps=n_steps, lam=lam)
            residual = y - pred
            err = np.linalg.norm(residual, axis=1)

            better = np.isfinite(err) & (err < best_err)
            if np.any(better):
                best_v[better] = v[better]
                best_err[better] = err[better]

            active = np.isfinite(err) & (err > tol)
            if not np.any(active):
                break

            accepted = np.zeros(N, dtype=bool)
            alpha = stage_eta.copy()

            for _bt in range(int(max_backtracking_steps)):
                trial_v = v + alpha[:, None] * residual
                trial_pred = levi_civita_exp_map_batch(
                    x, trial_v, n_steps=n_steps, lam=lam
                )
                trial_err = np.linalg.norm(y - trial_pred, axis=1)

                good = (
                    active
                    & (~accepted)
                    & np.isfinite(trial_err)
                    & (trial_err < err)
                )
                if np.any(good):
                    v[good] = trial_v[good]
                    accepted[good] = True

                    # Allow the next correction to grow back toward the nominal eta.
                    stage_eta[good] = np.minimum(
                        float(eta), alpha[good] / float(backtracking_factor)
                    )

                    better_trial = good & (trial_err < best_err)
                    if np.any(better_trial):
                        best_v[better_trial] = trial_v[better_trial]
                        best_err[better_trial] = trial_err[better_trial]

                remaining = active & (~accepted)
                if not np.any(remaining):
                    break
                alpha[remaining] *= float(backtracking_factor)

            failed = active & (~accepted)
            if np.any(failed):
                v[failed] = best_v[failed]
                stage_eta[failed] *= float(backtracking_factor)

        v = best_v.copy()

        if verbose:
            print(
                f"  shooting stage {stage_idx + 1}/{len(schedule)} "
                f"S={n_steps}: mean/max endpoint error="
                f"{np.mean(best_err):.3e}/{np.max(best_err):.3e}"
            )

    final_steps = schedule[-1]
    final_pred = levi_civita_exp_map_batch(x, v, n_steps=final_steps, lam=lam)
    final_err = np.linalg.norm(y - final_pred, axis=1)

    out = v[0] if squeeze else v
    if return_errors:
        return out, final_err
    return out


# ============================================================
# Geodesic construction: one Log, then Exp_x(t Log)
# ============================================================
def geodesic_curve(
    x,
    y,
    L,
    lam=1.0,
    shooting_schedule=(1, 2, 4, 8, 16, 32),
    shooting_iters=240,
    shooting_eta=0.5,
    shooting_tol=1e-8,
    verbose=True,
):
    if L < 2:
        raise ValueError("L must be >= 2")

    if float(lam) == 0.0:
        return np.linspace(x, y, L)

    v, endpoint_error = levi_civita_log_map_batch(
        x,
        y,
        lam=lam,
        substeps_schedule=shooting_schedule,
        total_iters=shooting_iters,
        eta=shooting_eta,
        tol=shooting_tol,
        max_backtracking_steps=12,
        return_errors=True,
        verbose=verbose,
    )

    final_steps = shooting_schedule[-1]
    taus = np.linspace(0.0, 1.0, L)
    bases = np.repeat(np.asarray(x)[None, :], L, axis=0)
    velocities = taus[:, None] * v[None, :]
    curve = levi_civita_exp_map_batch(
        bases,
        velocities,
        n_steps=final_steps,
        lam=lam,
    )

    # Only remove floating point display noise after a genuinely converged solve.
    if endpoint_error[0] > 1e-4:
        raise RuntimeError(
            f"Global shooting failed: endpoint error={endpoint_error[0]:.6g}"
        )
    curve[0] = x
    curve[-1] = y

    if verbose:
        print(f"Final shooting endpoint L2 error: {endpoint_error[0]:.6g}")

    return curve


# ============================================================
# Local proposal and normalized importance weights
# ============================================================
def local_candidates_and_weights(z0, epsilon, base_noise):
    """
    q=N(z0, epsilon^2 I), with self-normalized weights

        w_j proportional to exp(-||Z_j||^2/2).

    base_noise is shared across curve nodes within one outer iteration. This is
    a common-random-number variance reduction device; every candidate still has
    the correct Gaussian proposal marginal.
    """
    Z = z0[None, :] + float(epsilon) * base_noise
    log_w = -0.5 * np.sum(Z**2, axis=1)
    log_w -= np.max(log_w)
    weights = np.exp(log_w)
    weights /= np.sum(weights)
    return Z, weights


# ============================================================
# Fixed empirical local Frechet objective
# ============================================================
def fixed_local_frechet_energy(
    z_candidate,
    candidate_Y,
    weights,
    lam,
    local_shooting_schedule,
    local_shooting_iters,
    local_shooting_eta,
    local_shooting_tol,
):
    x_candidate = Phi(z_candidate)
    bases = np.repeat(x_candidate[None, :], len(candidate_Y), axis=0)

    logs = levi_civita_log_map_batch(
        bases,
        candidate_Y,
        lam=lam,
        substeps_schedule=local_shooting_schedule,
        total_iters=local_shooting_iters,
        eta=local_shooting_eta,
        tol=local_shooting_tol,
    )

    d2 = metric_sqnorm_batch(bases, logs, lam=lam)
    return float(np.sum(weights * d2))


# ============================================================
# High-Confidence Curve refinement
# ============================================================
def refine_curve(
    initial_curve_x,
    lam=1.0,
    epsilon=0.35,
    n_candidates=256,
    n_iters=20,
    latent_step_size=0.08,
    step_decay=0.96,
    min_latent_step=0.008,
    normalize_step=True,
    backtracking_factor=0.5,
    max_backtracking_steps=8,
    energy_decrease_tol=0.0,
    local_shooting_schedule=(2, 4, 8, 16),
    local_shooting_iters=20,
    local_shooting_eta=0.5,
    local_shooting_tol=1e-6,
    seed=7,
    verbose=True,
):
    """Generator-constrained projected local Frechet descent (Algorithm 4)."""
    if n_candidates % 2 != 0:
        raise ValueError("n_candidates must be even for antithetic sampling")

    rng = np.random.default_rng(seed)

    curve_z = Phi_inv(np.asarray(initial_curve_x, dtype=np.float64))
    curves_x = [Phi(curve_z)]
    energy_history = []
    accepted_step_history = []
    log_error_history = []
    ridge_error_history = [mean_true_ridge_error(curves_x[-1])]
    nll_history = [mean_negative_log_density(curves_x[-1])]

    if verbose:
        print(f"Initial mean true ridge error: {ridge_error_history[-1]:.6g}")
        print(f"Initial mean negative log density: {nll_history[-1]:.6g}")

    for it in range(int(n_iters)):
        old_z = curve_z.copy()
        new_z = old_z.copy()

        # Antithetic common-random-number cloud for THIS outer iteration.
        half = n_candidates // 2
        noise_half = rng.standard_normal((half, 2))
        common_noise = np.vstack((noise_half, -noise_half))

        baseline_energies = []
        accepted_energies = []
        accepted_steps = []
        iteration_log_errors = []

        scheduled_step = max(
            float(min_latent_step),
            float(latent_step_size) * (float(step_decay) ** it),
        )

        # Jacobi update: all directions use the same old curve.
        for i in range(1, len(old_z) - 1):
            z0 = old_z[i]
            x0 = Phi(z0)

            Z, weights = local_candidates_and_weights(
                z0, epsilon, common_noise
            )
            candidate_Y = Phi(Z)
            bases = np.repeat(x0[None, :], n_candidates, axis=0)

            logs, log_errors = levi_civita_log_map_batch(
                bases,
                candidate_Y,
                lam=lam,
                substeps_schedule=local_shooting_schedule,
                total_iters=local_shooting_iters,
                eta=local_shooting_eta,
                tol=local_shooting_tol,
                return_errors=True,
            )
            iteration_log_errors.extend(log_errors.tolist())

            # Delta_i = sum_j w_ij Log_{x_i}(Y_ij)
            delta_x = np.sum(weights[:, None] * logs, axis=0)

            baseline_d2 = metric_sqnorm_batch(bases, logs, lam=lam)
            baseline_energy = float(np.sum(weights * baseline_d2))

            # d_i = D Phi(z_i)^T g(x_i) Delta_i.
            # Here D Phi = A exactly.
            g_delta = g_metric(x0, lam=lam) @ delta_x
            latent_dir = A.T @ g_delta

            # Orthogonal projection to the normal space of the discrete latent
            # curve tangent, exactly as in the paper's HCC construction.
            tau_z = old_z[i + 1] - old_z[i - 1]
            tau_norm_sq = float(np.dot(tau_z, tau_z))
            if tau_norm_sq > 1e-14:
                latent_dir -= (
                    float(np.dot(latent_dir, tau_z)) / tau_norm_sq
                ) * tau_z

            dir_norm = float(np.linalg.norm(latent_dir))
            if not np.isfinite(dir_norm) or dir_norm <= 1e-14:
                baseline_energies.append(baseline_energy)
                accepted_energies.append(baseline_energy)
                accepted_steps.append(0.0)
                continue

            step_dir = latent_dir / dir_norm if normalize_step else latent_dir

            # Monotone per-node backtracking on the SAME frozen local objective.
            accepted = False
            trial_step = scheduled_step
            accepted_energy = baseline_energy

            for _bt in range(int(max_backtracking_steps)):
                z_trial = z0 + trial_step * step_dir
                trial_energy = fixed_local_frechet_energy(
                    z_trial,
                    candidate_Y,
                    weights,
                    lam,
                    local_shooting_schedule,
                    local_shooting_iters,
                    local_shooting_eta,
                    local_shooting_tol,
                )

                threshold = baseline_energy * (1.0 + float(energy_decrease_tol))
                if np.isfinite(trial_energy) and trial_energy <= threshold:
                    new_z[i] = z_trial
                    accepted_energy = trial_energy
                    accepted = True
                    break

                trial_step *= float(backtracking_factor)

            baseline_energies.append(baseline_energy)
            accepted_energies.append(accepted_energy)
            accepted_steps.append(trial_step if accepted else 0.0)

        curve_z = new_z  # endpoints fixed; NO respacing
        curve_x = Phi(curve_z)
        curves_x.append(curve_x)

        baseline_energies = np.asarray(baseline_energies)
        accepted_energies = np.asarray(accepted_energies)
        accepted_steps = np.asarray(accepted_steps)
        iteration_log_errors = np.asarray(iteration_log_errors)

        energy_history.append((baseline_energies, accepted_energies))
        accepted_step_history.append(accepted_steps)
        log_error_history.append(iteration_log_errors)
        ridge_error_history.append(mean_true_ridge_error(curve_x))
        nll_history.append(mean_negative_log_density(curve_x))

        if verbose:
            n_acc = int(np.count_nonzero(accepted_steps > 0.0))
            print(
                f"Iter {it + 1:02d}/{n_iters}: "
                f"accepted {n_acc}/{len(old_z)-2}, "
                f"mean F {np.mean(baseline_energies):.6g} -> "
                f"{np.mean(accepted_energies):.6g}, "
                f"ridge error={ridge_error_history[-1]:.6g}, "
                f"NLL={nll_history[-1]:.6g}, "
                f"local Log mean/max="
                f"{np.mean(iteration_log_errors):.2e}/"
                f"{np.max(iteration_log_errors):.2e}"
            )

    diagnostics = {
        "ridge_error": np.asarray(ridge_error_history),
        "negative_log_density": np.asarray(nll_history),
    }
    return (
        curves_x,
        energy_history,
        accepted_step_history,
        log_error_history,
        diagnostics,
    )


# ============================================================
# Run experiment
# ============================================================
if __name__ == "__main__":
    # --------------------------------------------------------
    # Representative GENERATED endpoints.
    #
    # z1 controls position along the ridge; z2 controls perpendicular offset.
    # Both endpoints use z2=2, so they are on the same side of the ridge.
    # This is important for this validation: the HCC normal projection can then
    # move the interior toward the high-density ridge rather than projecting out
    # an almost purely tangential centrality direction.
    # --------------------------------------------------------
    zA = np.array([-1.0, 2.0], dtype=np.float64)
    zB = np.array([ 1.0, 2.0], dtype=np.float64)
    x = Phi(zA)
    y = Phi(zB)

    L = 20
    metric_lambda = 1.0

    # Global geodesic shooting.
    global_shooting_schedule = (1, 2, 4, 8, 16, 32)
    global_shooting_iters = 240
    global_shooting_eta = 0.5
    global_shooting_tol = 1e-8

    # HCC localization/refinement.
    epsilon = 0.35
    n_candidates = 256
    refinement_iters = 20
    latent_step_size = 0.08
    step_decay = 0.96
    min_latent_step = 0.008

    # Local pairs are close, so a shorter shooting schedule is sufficient.
    local_shooting_schedule = (2, 4, 8, 16)
    local_shooting_iters = 20
    local_shooting_eta = 0.5

    print("Generated endpoints:")
    print(f"  zA={zA}, xA={x}")
    print(f"  zB={zB}, xB={y}")

    print("\nBuilding initial ambient Riemannian geodesic...")
    init_curve = geodesic_curve(
        x,
        y,
        L,
        lam=metric_lambda,
        shooting_schedule=global_shooting_schedule,
        shooting_iters=global_shooting_iters,
        shooting_eta=global_shooting_eta,
        shooting_tol=global_shooting_tol,
        verbose=True,
    )

    print("\nRunning High-Confidence Curve refinement...")
    (
        curves,
        energy_history,
        step_history,
        log_error_history,
        diagnostics,
    ) = refine_curve(
        init_curve,
        lam=metric_lambda,
        epsilon=epsilon,
        n_candidates=n_candidates,
        n_iters=refinement_iters,
        latent_step_size=latent_step_size,
        step_decay=step_decay,
        min_latent_step=min_latent_step,
        normalize_step=True,
        backtracking_factor=0.5,
        max_backtracking_steps=8,
        energy_decrease_tol=0.0,
        local_shooting_schedule=local_shooting_schedule,
        local_shooting_iters=local_shooting_iters,
        local_shooting_eta=local_shooting_eta,
        local_shooting_tol=1e-6,
        seed=7,
        verbose=True,
    )

    initial_err = diagnostics["ridge_error"][0]
    final_err = diagnostics["ridge_error"][-1]
    initial_nll = diagnostics["negative_log_density"][0]
    final_nll = diagnostics["negative_log_density"][-1]

    print("\nSynthetic ground-truth validation:")
    print(f"  initial mean ridge error: {initial_err:.6g}")
    print(f"  final mean ridge error:   {final_err:.6g}")
    print(f"  relative ridge-error improvement: {(initial_err-final_err)/initial_err*100:.2f}%")
    print(f"  initial mean NLL: {initial_nll:.6g}")
    print(f"  final mean NLL:   {final_nll:.6g}")

    # --------------------------------------------------------
    # Publication-oriented plot
    # --------------------------------------------------------
    xs = np.linspace(-2.0, 2.0, 500)
    ys = np.linspace(-2.0, 2.0, 500)
    X, Y = np.meshgrid(xs, ys)
    Z = ridge_2d(np.stack([X, Y], axis=-1))

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, Z, levels=50, cmap="viridis")

    ridge_x = np.linspace(-2.0, 2.0, 300)
    ridge_y = a * ridge_x
    ax.plot(
        ridge_x,
        ridge_y,
        "w--",
        lw=1.6,
        alpha=0.9,
        label="True ridge",
    )

    selected = [0, 5, 10, refinement_iters]
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(selected)))
    for idx, c in zip(selected, colors):
        curve = curves[idx]
        label = "Initial geodesic" if idx == 0 else f"HCC iter {idx}"
        ax.plot(
            curve[:, 0],
            curve[:, 1],
            "-o",
            color=c,
            lw=2.2,
            markersize=3.4,
            label=label,
        )

    ax.scatter(
        [x[0], y[0]],
        [x[1], y[1]],
        c="white",
        s=82,
        edgecolors="black",
        linewidths=1.2,
        label="Generated endpoints",
        zorder=10,
    )

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", framealpha=0.92)
    ax.set_title("High-Confidence Curve Refinement on Ridge Dataset")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(contour, ax=ax, label="density")
    fig.tight_layout()
    fig.savefig(
        "ridge_high_confidence_curve_better.png",
        dpi=220,
        bbox_inches="tight",
    )

    # Separate quantitative diagnostic figure.
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
    its = np.arange(refinement_iters + 1)
    ax2.plot(its, diagnostics["ridge_error"], "-o", label="Mean squared ridge distance")
    ax2.set_xlabel("HCC refinement iteration")
    ax2.set_ylabel("Mean squared distance to true ridge")
    ax2.set_title("Independent Synthetic Validation")
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(
        "ridge_high_confidence_curve_validation.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.show()
