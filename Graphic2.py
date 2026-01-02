"""
Graphic2.py
Alexander Marsh
2 January 2026

Displays expected primitive updates in the simple Gaussian case.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 2D Standard Gaussian
# ============================================================
def gaussian_2d(x):
    return (1.0 / (2 * np.pi)) * np.exp(-0.5 * np.sum(x**2, axis=-1))

def score(x):
    """∇ log p(x) for N(0, I)"""
    return -x-np.log(2*np.pi)

# ============================================================
# Inverse Riemannian metric (Algorithm 2)
# ============================================================
def g_inv_metric(x, theta=0.0):
    eps = 1e-300
    p_x = np.maximum(gaussian_2d(x), eps)
    p_ind = gaussian_2d(np.zeros_like(x))
    s = score(x)
    w = np.exp(-theta * (np.log(p_x) - np.log(p_ind)))
    denom = 1.0 + w * np.dot(s, s)
    return np.eye(len(x)) - (w / denom) * np.outer(s, s)

# ============================================================
# First-order Riemannian exponential map (Algorithm 2)
# ============================================================
def riemannian_exponential_first_order(x, v, n_steps, theta=0.0):
    x_k = x.copy()
    dv = v / n_steps
    for _ in range(n_steps):
        ginv = g_inv_metric(x_k, theta=theta)
        x_k = x_k + ginv @ dv

    return x_k

# ============================================================
# Riemannian logarithm map via shooting (Algorithm 1)
# ============================================================
def riemannian_logarithm(x, y, n_steps, theta=0.0, eta=1e-3, n_max_iter=300, tol=1e-3):
    v = np.zeros_like(x)
    for _ in range(n_max_iter):
        exp_v = riemannian_exponential_first_order(x, v, n_steps, theta)
        residual = y - exp_v
        loss = np.linalg.norm(residual) ** 2
        if loss < tol:
            break
        v += eta * residual

    return v

# ============================================================
# Gradient descent for expected primitive (Algorithm 3)
# ============================================================
def expected_primitive(samples, z0, theta=0.0, eta=5e-3, Q=20, n_steps=10,):
    z = z0.copy()
    trajectory = [z.copy()]
    for _ in range(Q):
        grad = np.zeros_like(z)
        for zi in samples:
            L = riemannian_logarithm(z, zi, n_steps, theta=theta)
            weight = np.exp(0.5 * (np.dot(z, z) - np.dot(zi, zi)))
            grad += -2.0 * weight * L
        z = z - eta * grad
        trajectory.append(z.copy())

    return np.array(trajectory)

if __name__ == "__main__":
    rng = np.random.default_rng(5)

    # Sample points zi ~ μ
    N = 20
    samples = rng.normal(size=(N, 2))

    # Random initialization
    z0 = rng.uniform(-2, 2, size=2)

    # Parameters
    theta = 1.0
    Q = 20
    n_steps = 10

    traj = expected_primitive(samples, z0, theta=theta, eta=5e-3, Q=Q, n_steps=n_steps,)

    # ========================================================
    # Plot
    # ========================================================
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = gaussian_2d(np.stack([X, Y], axis=-1))

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=40, cmap="viridis")

    plt.plot(traj[:, 0], traj[:, 1], "-o", color="red", lw=2, label="Expected primitive updates",)

    plt.scatter(z0[0], z0[1], c="yellow", s=80, label="Start")
    plt.scatter(traj[-1, 0], traj[-1, 1], c="cyan", s=80, label="Final",)

    plt.legend()
    plt.title("Gradient Descent Toward Expected Primitive")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Gaussian density")
    plt.show()
