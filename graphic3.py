import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 2D Standard Gaussian
# ============================================================
def gaussian_2d(x):
    return (1.0 / (2 * np.pi)) * np.exp(-0.5 * np.sum(x**2, axis=-1))


def score(x):
    return -x


def confidence(x):
    return 0.5 * np.dot(x, x)


# ============================================================
# Metric (from SECOND code)
# ============================================================
def g_metric(x):
    eps = 1e-300
    p_x = np.maximum(gaussian_2d(x), eps)
    p_ind = gaussian_2d(np.zeros(2))

    s = -x - np.log(2 * np.pi)
    weight = np.exp(np.log(p_x) - np.log(p_ind))

    return np.eye(2) + weight * np.outer(s, s)


# ============================================================
# Riemannian exponential (from SECOND code)
# ============================================================
def riemannian_exponential(x, v, n_steps):
    xk = x.copy()
    dv = v / n_steps

    for _ in range(n_steps):
        g = g_metric(xk)
        xk = xk + np.linalg.solve(g, dv)

    return xk


# ============================================================
# Riemannian logarithm (from SECOND code)
# ============================================================
def riemannian_logarithm(x, y, n_steps, eta=1e-3, n_iters=1000):
    v = np.zeros_like(x)

    for _ in range(n_iters):
        exp_v = riemannian_exponential(x, v, n_steps)
        r = y - exp_v
        if np.linalg.norm(r) < 1e-3:
            break
        v += eta * r

    return v


# ============================================================
# Geodesic construction (SECOND CODE STYLE)
# ============================================================
def geodesic_curve(x, y, L):
    geo = [x.copy()]
    x_curr = x.copy()

    for i in range(1, L):
        v = riemannian_logarithm(x_curr, y, L)
        x_curr = riemannian_exponential(x_curr, v * (i / L), L)
        geo.append(x_curr)

    geo.append(y.copy())
    return np.array(geo)


# ============================================================
# High-confidence curve refinement (UNCHANGED)
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

    # Initial geodesic (SECOND CODE)
    init_curve = geodesic_curve(x, y, L)

    # Refinement (FIRST CODE)
    curves = refine_curve(init_curve)

    # ========================================================
    # Plot
    # ========================================================
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
    plt.title("High-Confidence Interpolation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Gaussian density")
    plt.show()
