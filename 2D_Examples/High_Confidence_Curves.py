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

# Unit normal to the ridge
n = np.array([-a, 1.0])
n = n / np.linalg.norm(n)

# Unit tangent to the ridge
t_vec = np.array([1.0, a])
t_vec = t_vec / np.linalg.norm(t_vec)

# ============================================================
# Utility: Arc-length resampling
# ============================================================
def resample_by_arclength(curve, L):
    diffs = np.diff(curve, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)

    s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    if s[-1] == 0:
        return curve[:L].copy()

    s /= s[-1]
    s_uniform = np.linspace(0, 1, L)

    new_curve = np.zeros((L, curve.shape[1]))
    for d in range(curve.shape[1]):
        new_curve[:, d] = np.interp(s_uniform, s, curve[:, d])

    return new_curve

# ============================================================
# Stable ridge density
# ============================================================
def ridge_2d(x):
    x = np.atleast_2d(x)

    x_par = x @ t_vec
    x_perp = x @ n

    exponent = -0.5 * (
        x_par**2 / sigma_par**2 +
        x_perp**2 / sigma_perp**2
    )

    # prevent overflow/underflow in exp
    exponent = np.clip(exponent, -700, 50)

    return np.exp(exponent)

# ============================================================
# Score function
# ============================================================
def score(x):
    x_par = np.dot(x, t_vec)
    x_perp = np.dot(x, n)

    grad_par = -(x_par / sigma_par**2) * t_vec
    grad_perp = -(x_perp / sigma_perp**2) * n

    s = grad_par + grad_perp

    # clip extreme score values
    s_norm = np.linalg.norm(s)
    if s_norm > 1e3:
        s = s / s_norm * 1e3

    return s

# ============================================================
# Confidence (distance to ridge)
# ============================================================
def confidence(x):
    x_perp = np.dot(x, n)
    return x_perp**2

# ============================================================
# Metric
# ============================================================
def g_metric(x):
    eps = 1e-12
    p_x = ridge_2d(x)
    p0 = ridge_2d(np.zeros(2))

    weight = np.clip(p_x / (p0 + eps), 0.0, 1.0)

    s = score(x)

    return np.eye(2) + weight * np.outer(s, s)

# ============================================================
# Jacobian-vector product
# ============================================================
def Jv(x, v):
    J = (
        -(1 / sigma_par**2) * np.outer(t_vec, t_vec)
        -(1 / sigma_perp**2) * np.outer(n, n)
    )
    return J @ v

# ============================================================
# Levi-Civita exponential map
# ============================================================
def levi_civita_exp_map(x, v, n_steps=10, lam=1.0):
    x_curr = x.copy()
    v_curr = v.copy()
    dt = 1.0 / n_steps

    for _ in range(n_steps):
        s = score(x_curr)
        g = g_metric(x_curr)

        Jv_score = Jv(x_curr, v_curr)
        inner = np.dot(v_curr, Jv_score)
        s_norm2 = np.dot(s, s)

        denom = 1.0 + lam * s_norm2
        denom = max(denom, 1e-8)

        accel = -lam * inner / denom * s

        v_curr = v_curr + dt * accel
        dx = np.linalg.solve(g, v_curr)
        x_curr = x_curr + dx * dt

    return x_curr

# ============================================================
# Levi-Civita logarithm map
# ============================================================
def levi_civita_log_map(x, y, n_steps=10, eta=1e-3, n_iters=200, lam=1.0):
    v = np.zeros_like(x)

    for _ in range(n_iters):
        exp_v = levi_civita_exp_map(x, v, n_steps=n_steps, lam=lam)
        r = y - exp_v

        if np.linalg.norm(r) < 1e-5:
            break

        v += eta * r

        # 🔑 critical stabilization: clamp velocity
        v_norm = np.linalg.norm(v)
        if v_norm > 5.0:
            v = v / v_norm * 5.0

    return v

# ============================================================
# Geodesic construction
# ============================================================
def geodesic_curve(x, y, L, n_steps=10, lam=1.0):
    geo = [x.copy()]
    x_curr = x.copy()

    for i in range(1, L):
        tau = i / (L - 1)
        v = levi_civita_log_map(x_curr, y, n_steps=n_steps, lam=lam)
        x_curr = levi_civita_exp_map(x_curr, tau * v, n_steps=n_steps, lam=lam)
        geo.append(x_curr.copy())

    geo.append(y.copy())
    return np.array(geo)

# ============================================================
# High-confidence curve refinement
# ============================================================
def refine_curve(curve, xi=0.01, n_iters=6, L_resample=10):
    curves = []

    curve = resample_by_arclength(curve, L_resample)
    curves.append(curve.copy())

    for _ in range(n_iters):
        new_curve = curve.copy()

        for i in range(1, len(curve) - 1):
            x0 = curve[i]
            candidates = x0 + xi * np.random.randn(50, 2)
            costs = np.array([confidence(c) for c in candidates])
            new_curve[i] = candidates[np.argmin(costs)]

        curve = resample_by_arclength(new_curve, L_resample)
        curves.append(curve.copy())

    return curves

# ============================================================
# Run experiment
# ============================================================
if __name__ == "__main__":
    x = np.array([-1.5, 1.0])
    y = np.array([1.5, -1.0])
    L = 20

    init_curve = geodesic_curve(x, y, L)
    curves = refine_curve(init_curve)

    xs = np.linspace(-2, 2, 400)
    ys = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = ridge_2d(np.stack([X, Y], axis=-1))

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=40, cmap="viridis")

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(curves)))
    for k, (curve, c) in enumerate(zip(curves, colors)):
        plt.plot(curve[:, 0], curve[:, 1], "-o", color=c, lw=2, label=f"Iter {k}")

    plt.scatter([x[0], y[0]],
                [x[1], y[1]],
                c="white", s=80, edgecolors="black",
                label="Endpoints")

    plt.legend()
    plt.title("Levi-Civita Geodesic on Ridge Dataset (Stable)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Ridge density")
    plt.show()
