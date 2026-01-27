"""
Graphic.py
Alexander Marsh
27 January 2026

Displays geodesics with our metric in the simple Gaussian case for various values of theta
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Gaussian + log-density
# -----------------------------
def gaussian_2d(x):
    return (1/(2*np.pi)) * np.exp(-0.5*np.sum(x**2, axis=-1))

def log_gaussian_2d(x):
    return -0.5*np.sum(x**2, axis=-1) - np.log(2*np.pi)

# -----------------------------
# Weight w(x) = exp(theta (log p(x) - log p(0)))
# -----------------------------
def weight(x, theta):
    return theta#np.exp(theta * (log_gaussian_2d(x) - log_gaussian_2d(np.zeros(2))))

# -----------------------------
# Geodesic acceleration
# -----------------------------
def geodesic_acceleration(x, v, theta):
    w = weight(x, theta)

    r2 = np.dot(x, x)
    v2 = np.dot(v, v)
    xv = np.dot(x, v)

    coeff = w / (1.0 + w * r2)

    return -coeff * (v2 * x - xv * v)

# -----------------------------
# One symplectic geodesic step
# -----------------------------
def geodesic_step(x, v, dt, theta):
    a = geodesic_acceleration(x, v, theta)
    v_new = v + dt * a
    x_new = x + dt * v_new
    return x_new, v_new

# -----------------------------
# Riemannian exponential map
# -----------------------------
def riemannian_exponential(x0, v0, n_steps, theta, dt=1e-2):
    x = x0.copy()
    v = v0.copy()

    for _ in range(n_steps):
        x, v = geodesic_step(x, v, dt, theta)

    return x

# -----------------------------
# Shooting method for log map
# -----------------------------
def riemannian_logarithm(
    x, y, theta,
    n_steps=200,
    dt=1e-2,
    lr=1e-2,
    n_iters=2000,
    tol=1e-3
):
    v = np.zeros_like(x)

    for i in range(n_iters):
        exp_v = riemannian_exponential(x, v, n_steps, theta, dt)
        residual = y - exp_v
        loss = np.linalg.norm(residual)**2

        if loss < tol:
            break

        # Simple shooting gradient (Euclidean)
        v += lr * residual

        if i % 100 == 0:
            print(f"Iter {i}, loss {loss:.6f}")

    return v

# -----------------------------
# Setup
# -----------------------------
start = np.array([-1.0, -1.0])
end   = np.array([ 1.0, 0.0])

theta_values = [-5.0, -2.0, 0.0, 2.0]
colors = ["red", "blue", "green", "orange"]

# -----------------------------
# Plot Gaussian contours
# -----------------------------
xg = np.linspace(-3, 3, 200)
yg = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(xg, yg)
Z = gaussian_2d(np.stack([X, Y], axis=-1))

plt.figure(figsize=(8,6))
plt.contourf(X, Y, Z, levels=40, cmap="viridis")
plt.colorbar(label="Gaussian density")

# -----------------------------
# Plot geodesics
# -----------------------------
for theta, color in zip(theta_values, colors):
    v0 = riemannian_logarithm(start, end, theta)
    path = [start]

    x = start.copy()
    v = v0.copy()

    for _ in range(200):
        x, v = geodesic_step(x, v, dt=1e-2, theta=theta)
        path.append(x.copy())

    path = np.array(path)

    plt.plot(path[:,0], path[:,1], color=color, lw=2, label=f"theta={theta}")
    plt.scatter(path[:,0], path[:,1], color=color, s=10)

plt.scatter([start[0], end[0]], [start[1], end[1]],
            color="white", s=60, label="Endpoints")

plt.legend()
plt.title("True Riemannian Geodesics for g = I + w(x) s sᵀ")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

