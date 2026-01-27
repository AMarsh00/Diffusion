"""
Graphic3.py
Alexander Marsh
27 January 2026

Displays the high-confidence-curve example for the simple Gaussian case.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 2D Standard Gaussian
# ============================================================
def gaussian_2d(x):
    return (1.0 / (2 * np.pi)) * np.exp(-0.5 * np.sum(x**2, axis=-1))


def score(x):
    return -x  # grad log p(x) for standard Gaussian


def confidence(x):
    return 0.5 * np.dot(x, x)  # squared norm as "energy"

# ============================================================
# Metric (Levi-Civita inspired)
# ============================================================
def g_metric(x):
    """
    Metric G(x) = I + s s^T * weight
    where s = score(x), weight ~ p(x)/p(0)
    """
    eps = 1e-300
    p_x = np.maximum(gaussian_2d(x), eps)
    p0 = gaussian_2d(np.zeros(2))
    weight = p_x / p0
    s = score(x)
    return np.eye(2) + weight * np.outer(s, s)


# ============================================================
# Jacobian-vector product for Levi-Civita
# ============================================================
def Jv(x, v):
    """
    Compute Jv = d(score)/dx @ v = Jacobian-vector product
    For Gaussian: score = -x, so J = -I
    """
    return -v  # simple for 2D Gaussian


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

        # acceleration along Levi-Civita connection
        Jv_score = Jv(x_curr, v_curr)
        s_flat = s
        v_flat = v_curr
        s_norm2 = np.dot(s_flat, s_flat)
        inner = np.dot(v_flat, Jv_score)

        accel = -lam * inner / (1 + lam * s_norm2) * s

        # update velocity and position
        v_curr = v_curr + dt * accel
        # solve G dx = dv for dx
        dx = np.linalg.solve(g, v_curr)
        x_curr = x_curr + dx * dt

    return x_curr


# ============================================================
# Levi-Civita logarithm map
# ============================================================
def levi_civita_log_map(x, y, n_steps=10, eta=1e-2, n_iters=500, lam=1.0):
    v = np.zeros_like(x)

    for _ in range(n_iters):
        exp_v = levi_civita_exp_map(x, v, n_steps=n_steps, lam=lam)
        r = y - exp_v
        if np.linalg.norm(r) < 1e-5:
            break
        v += eta * r

    return v


# ============================================================
# Geodesic construction (Levi-Civita style)
# ============================================================
def geodesic_curve(x, y, L, n_steps=10, lam=1.0):
    geo = [x.copy()]
    x_curr = x.copy()

    for i in range(1, L):
        t = i / (L - 1)
        v = levi_civita_log_map(x_curr, y, n_steps=n_steps, lam=lam)
        x_curr = levi_civita_exp_map(x_curr, t * v, n_steps=n_steps, lam=lam)
        geo.append(x_curr.copy())

    geo.append(y.copy())
    return np.array(geo)


# ============================================================
# High-confidence curve refinement (unchanged)
# ============================================================
def refine_curve(curve, xi=0.01, n_iters=6):
    curves = [curve.copy()]

    for _ in range(n_iters):
        new_curve = curve.copy()

        for i in range(1, len(curve) - 1):
            x0 = curve[i]
            candidates = x0 + xi * np.random.randn(50, 2)
            costs = np.array([confidence(c) for c in candidates])
            new_curve[i] = candidates[np.argmin(costs)]

        curve = new_curve
        curves.append(curve.copy())

    return curves


# ============================================================
# Run experiment
# ============================================================
if __name__ == "__main__":
    x = np.array([1.0, 2.0])
    y = np.array([1.0, -2.0])
    L = 20

    # Initial geodesic (Levi-Civita)
    init_curve = geodesic_curve(x, y, L, n_steps=10, lam=1.0)

    # Refinement (high-confidence)
    curves = refine_curve(init_curve)

    # ============================================================
    # Plot
    # ============================================================
    xs = np.linspace(-2, 2, 300)
    ys = np.linspace(-2, 2, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = gaussian_2d(np.stack([X, Y], axis=-1))

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=40, cmap="viridis")

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(curves)))

    for k, (curve, c) in enumerate(zip(curves, colors)):
        plt.plot(curve[:, 0], curve[:, 1], "-o", color=c, lw=2, label=f"Iter {k}")

    plt.scatter([x[0], y[0]], [x[1], y[1]],
                c="white", s=80, edgecolors="black", label="Endpoints")

    plt.legend()
    plt.title("Levi-Civita Geodesic with High-Confidence Refinement")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Gaussian density")
    plt.show()
