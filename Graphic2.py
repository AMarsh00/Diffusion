"""
Graphic2.py
Alexander Marsh
27 January 2026

Displays expected primitive updates in the simple Gaussian case.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 2D Standard Gaussian
# ============================================================
def gaussian_2d(x):
    return (1.0 / (2 * np.pi)) * np.exp(-0.5 * np.sum(x**2, axis=-1))


def log_gaussian_2d(x):
    return -0.5 * np.sum(x**2, axis=-1) - np.log(2 * np.pi)


def score(x):
    """∇ log p(x) for N(0, I)"""
    return -x


# ============================================================
# Weight w(x) = exp(theta (log p(x) - log p(0)))
# ============================================================
def weight(x, theta):
    return np.exp(theta * (log_gaussian_2d(x) - log_gaussian_2d(np.zeros_like(x))))


# ============================================================
# Geodesic acceleration (Christoffel contraction)
# ============================================================
def geodesic_acceleration(x, v, theta):
    w = weight(x, theta)

    r2 = np.dot(x, x)
    v2 = np.dot(v, v)
    xv = np.dot(x, v)

    coeff = w / (1.0 + w * r2)

    return -coeff * (v2 * x - xv * v)


# ============================================================
# One symplectic geodesic step
# ============================================================
def geodesic_step(x, v, dt, theta):
    a = geodesic_acceleration(x, v, theta)
    v_new = v + dt * a
    x_new = x + dt * v_new
    return x_new, v_new


# ============================================================
# Riemannian exponential via geodesic integration
# ============================================================
def riemannian_exponential(x0, v0, n_steps, theta, dt=1e-2):
    x = x0.copy()
    v = v0.copy()

    for _ in range(n_steps):
        x, v = geodesic_step(x, v, dt, theta)

    return x


# ============================================================
# Riemannian logarithm via shooting (updated)
# ============================================================
def riemannian_logarithm(
    x,
    y,
    theta=0.0,
    n_steps=200,
    dt=1e-2,
    eta=1e-2,
    n_max_iter=500,
    tol=1e-3,
):
    v = np.zeros_like(x)

    for _ in range(n_max_iter):
        exp_v = riemannian_exponential(x, v, n_steps, theta, dt)
        residual = y - exp_v
        loss = np.linalg.norm(residual) ** 2

        if loss < tol:
            break

        # Euclidean shooting update
        v += eta * residual

    return v


# ============================================================
# Gradient descent for expected primitive (UPDATED)
# ============================================================
def expected_primitive(
    samples,
    z0,
    theta=0.0,
    eta=5e-3,
    Q=20,
    n_steps=200,
    dt=1e-2,
):
    z = z0.copy()
    trajectory = [z.copy()]

    for _ in range(Q):
        grad = np.zeros_like(z)

        for zi in samples:
            L = riemannian_logarithm(
                z,
                zi,
                theta=theta,
                n_steps=n_steps,
                dt=dt,
            )
            weight_term = np.exp(0.5 * (np.dot(z, z) - np.dot(zi, zi)))
            grad += -2.0 * weight_term * L

        z = z - eta * grad
        trajectory.append(z.copy())

    return np.array(trajectory)


# ============================================================
# Run experiment
# ============================================================
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

    traj = expected_primitive(
        samples,
        z0,
        theta=theta,
        eta=5e-3,
        Q=Q,
        n_steps=200,
        dt=1e-2,
    )

    # ========================================================
    # Plot
    # ========================================================
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = gaussian_2d(np.stack([X, Y], axis=-1))

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=40, cmap="viridis")

    plt.plot(
        traj[:, 0],
        traj[:, 1],
        "-o",
        color="red",
        lw=2,
        label="Expected primitive (geodesic)",
    )

    plt.scatter(z0[0], z0[1], c="yellow", s=80, label="Start")
    plt.scatter(
        traj[-1, 0],
        traj[-1, 1],
        c="cyan",
        s=80,
        label="Final",
    )

    plt.legend()
    plt.title("Gradient Descent Toward Expected Primitive (True Geodesics)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Gaussian density")
    plt.show()

