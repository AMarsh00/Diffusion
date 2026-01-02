"""
Graphic.py
Alexander Marsh
2 January 2026

Displays geodesics with our metric in the simple Gaussian case for various values of theta
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Example 2D Gaussian
# -----------------------------
def gaussian_2d(x):
    """Standard 2D Gaussian pdf"""
    return (1/(2*np.pi)) * np.exp(-0.5*np.sum(x**2, axis=-1))

# -----------------------------
# Metric g(x) = I + e^{theta (log p(x) - log p_ind(x))} * s s^T
# -----------------------------
def g_metric(x, theta=0.0):
    """Compute the metric g(x) based on Gaussian distribution"""
    eps = 1e-300
    p_sharp = np.maximum(gaussian_2d(x), eps)
    p_sharp_ind = gaussian_2d(np.zeros(2))  # log pdf of N(0,I) at 0
    
    # Gradient of log p(x) with respect to x
    s = -x-np.log(2*np.pi)  # grad log p(x) for standard Gaussian
    
    # Compute weight based on the difference in log densities
    weight = np.exp(theta * (np.log(p_sharp) - np.log(p_sharp_ind)))

    # The metric is a scaled identity matrix with an additional term
    g = np.eye(2)+weight*np.outer(s,s)
    return g

# -----------------------------
# Compute Riemannian exponential map (using the metric g(x))
# -----------------------------
def riemannian_exponential(x, v, n_substeps, lam=1e6, theta=0.0, eps=1e-6):
    """Approximate the Riemannian exponential using g(x)"""
    x_current = x
    dx = v / n_substeps  # Split the tangent vector into smaller steps
    
    for step in range(n_substeps):
        # Compute the metric g(x)
        g = g_metric(x_current, lam=lam, theta=theta)
        
        # Check if g is singular (det(g) is close to zero)
        det_g = np.linalg.det(g)
        if abs(det_g) < eps:
            # Use identity matrix if g is singular or near singular
            print(f"Warning: g(x) is singular at step {step}, using identity matrix.")
            g = np.eye(len(x))  # Use identity matrix instead of g(x)
        
        # Perform the update using the inverse of g (or identity if singular)
        x_current = x_current + np.linalg.inv(g) @ dx  # Update the position
    
    return x_current

# -----------------------------
# Loss function for Riemannian logarithm
# -----------------------------
def loss(v, x, y, lam, n_substeps, theta=0.0):
    """Compute the squared Euclidean loss between exp_x(v) and target y"""
    exp_v = riemannian_exponential(x, v, n_substeps, lam=lam, theta=theta)
    return np.sum((exp_v - y) ** 2)

# -----------------------------
# Algorithm 1: Riemannian Logarithm Map via Shooting Method
# -----------------------------
def riemannian_logarithm(x, y, n_substeps, lam=1e6, theta=0.0, eta=1e-3, n_max_iter=10000, momentum_gamma=0.9, tol=1e-2):
    """Compute the Riemannian logarithm of y with respect to x using a momentum-based update."""
    
    v = np.zeros_like(x)  # Initialize tangent vector
    v = v.astype(np.float32)  # Ensure correct type for numerical gradients
    
    # Initialize momentum
    momentum = np.zeros_like(v)
    best_v = v.copy()  # Keep track of the best tangent vector
    best_loss = float('inf')  # Best loss seen so far

    n_substeps_schedule=[n_substeps//8, n_substeps//4, n_substeps//2, n_substeps]
    
    # Iterate over the gradient descent process
    for idx, n_substeps_ in enumerate(n_substeps_schedule):
        n_iters = n_max_iter // len(n_substeps_schedule)

        for i in range(n_iters):
            # Compute the current prediction using the Riemannian exponential map
            exp_v = riemannian_exponential(x, v, n_substeps_, lam=lam, theta=theta)
            
            # Residual (difference between predicted and target)
            residual = y - exp_v
            
            # Compute the loss (squared norm of the residual)
            loss = np.sum(residual**2)
            
            # If the loss is below the tolerance, break out of the loop
            if loss < tol:
                break
            
            # Normalize residual to avoid large updates
            residual_norm = np.linalg.norm(residual) + 1e-8  # To avoid division by zero
            step = eta * residual / residual_norm  # Compute the update step
            
            # Apply momentum update (exponentially weighted sum of previous gradients)
            momentum = momentum_gamma * momentum + step
            v += momentum  # Update the tangent vector
            
            # Track the best tangent vector at the last substeps stage
            if loss < best_loss:
                best_loss = loss
                best_v = v.copy()
            
            print(f"Iteration {idx*n_iters+i+1}/{n_max_iter}, Loss: {loss:.6f}")  # For debugging
    
    return best_v

# -----------------------------
# Random start/end points
# -----------------------------
rng = np.random.default_rng()
start = np.array([-1,-1])#rng.uniform(-3, 3, size=2)
end = np.array([1,1])#rng.uniform(-3, 3, size=2)

# -----------------------------
# Plot Gaussian contour
# -----------------------------
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = gaussian_2d(np.stack([X, Y], axis=-1))

plt.figure(figsize=(8,6))
plt.contourf(X, Y, Z, levels=30, cmap='viridis')

theta_values = [-5, -2.5, -1, -0.5, 0, 0.5, 1, 2.5, 5]  # Exploring different theta values
n_steps = 10  # Number of geodesic steps

colors = [
    'red', 'blue', 'orange', 'magenta', 'cyan', 'green', 'purple', 'yellow', 'pink', 'brown',
    'lime', 'teal', 'violet', 'indigo', 'gold', 'silver', 'coral', 'plum', 'azure', 'maroon',
    'turquoise', 'orchid', 'salmon', 'crimson', 'khaki', 'peachpuff', 'skyblue', 'seashell',
    'saddlebrown', 'darkgreen', 'royalblue', 'lavender', 'goldenrod', 'darkorange', 'fuchsia',
    'chartreuse', 'deepskyblue', 'lightgreen', 'mediumvioletred', 'slategray', 'rosybrown',
    'forestgreen', 'mediumslateblue', 'darkviolet', 'mediumseagreen', 'darkkhaki', 'lightcoral',
    'darkslateblue', 'mediumturquoise', 'lightskyblue', 'lightgoldenrodyellow', 'mediumorchid',
    'steelblue', 'darkred', 'darksalmon', 'lightseagreen', 'wheat', 'midnightblue', 'lightblue'
]
for theta, color in zip(theta_values, colors):
    # Compute the Riemannian logarithm with scaling
        
    # Approximate geodesic path by computing the Riemannian exponential
    geo = [start]
    for i in range(1, n_steps):
        # Recompute the logarithm after each step
        v_new = riemannian_logarithm(geo[-1], end, n_steps, theta=theta, eta=1e-3, n_max_iter=1000)
        geo.append(riemannian_exponential(geo[-1], v_new*i/n_steps, n_steps, theta=theta))
    geo.append(end)
    
    geo = np.array(geo)
    plt.plot(geo[:,0], geo[:,1], color=color, lw=2, label=f'theta={theta}')
    plt.scatter(geo[:,0], geo[:,1], color=color, s=20)  # Show points

plt.scatter([start[0], end[0]], [start[1], end[1]], color='white', s=50, label='Endpoints')
plt.title("Riemannian Logarithms and Geodesics with Step-wise Updates")
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Gaussian density')
plt.legend()
plt.show()

