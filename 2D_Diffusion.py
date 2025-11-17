import os
import math
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.stats import norm
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# 1?Toy 2D Curve Dataset
# ----------------------------
class Curve2DDataset(Dataset):
    def __init__(self, n_points=500):
        t = torch.linspace(0, 2 * torch.pi, n_points)
        r = 1.0 + 0.2 * torch.sin(5*t)
        x = r * torch.cos(t)
        y = r * torch.sin(t)
        self.data = torch.stack([x, y], dim=1)

        #self.data = torch.stack([torch.cos(t), torch.sin(t)], dim=1)  # Circle
        #self.data += 0.02 * torch.randn_like(self.data)  # optional noise

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

# ----------------------------
# 2?Sinusoidal Time Embeddings
# ----------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timestep):
        device = timestep.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timestep[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings

# Residual MLP block
class ResidualBlockMLP(nn.Module):
    def __init__(self, dim_in, dim_out, time_emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_out)
        self.fc2 = nn.Linear(dim_out, dim_out)
        self.time_mlp = nn.Linear(time_emb_dim, dim_out)
        self.act = nn.ReLU()
        if dim_in != dim_out:
            self.skip = nn.Linear(dim_in, dim_out)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t):
        h = self.act(self.fc1(x))
        h = h + self.time_mlp(t)
        h = self.act(self.fc2(h))
        return h + self.skip(x)

# 2D diffusion model
class MLPDiffusion2D(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, time_emb_dim=128, n_layers=4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim*2),
            nn.ReLU(),
            nn.Linear(time_emb_dim*2, time_emb_dim)
        )
        layers = []
        for i in range(n_layers):
            dim_in = input_dim if i == 0 else hidden_dim
            layers.append(ResidualBlockMLP(dim_in, hidden_dim, time_emb_dim))
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        for layer in self.layers:
            x = layer(x, t_emb)
        return self.out(x)
        
# ------------------------
# VP Scheduler
# ------------------------
class VPScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        alphas_cumprod = self.alphas_cumprod.to(x0.device)
        if isinstance(t, int):
            t = torch.tensor([t], device=x0.device)
        sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t]).view(-1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

def ddim_sample(model, scheduler, x_T, timesteps, eta=0.0):
    x_t = x_T
    device = x_t.device
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        alpha_t = alphas_cumprod[t]
        alpha_t_prev = alphas_cumprod[t_prev]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        t_tensor = torch.tensor([t], device=device)
        epsilon_theta = model(x_t, t_tensor)
        x0_pred = (x_t - sqrt_one_minus_alpha_t * epsilon_theta) / sqrt_alpha_t
        sigma_t = eta * torch.sqrt(
            (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
        )
        sigma_t_val = sigma_t.item()
        noise = torch.randn_like(x_t) if sigma_t_val > 0 else torch.zeros_like(x_t)
        x_t = sqrt_alpha_t_prev * x0_pred + torch.sqrt(1 - alpha_t_prev - sigma_t_val**2) * epsilon_theta + sigma_t * noise
    return x_t

# ------------------------
# Stein Score / Riemannian metric
def score_fn(x, model, scheduler, t):
    """
    Computes the Stein score s_theta(x, t) = grad_x log p_t(x)
    using epsilon model output.
    x: [B, C, H, W]
    t: int or tensor of shape [B]
    """
    device = x.device
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    if isinstance(t, int):
        t_tensor = torch.tensor([t], device=device)
        alpha_bar = alphas_cumprod[t]
    else:
        t_tensor = t
        alpha_bar = alphas_cumprod[t]

    epsilon_theta = model(x, t_tensor)  # [B,C,H,W]
    score = - epsilon_theta / torch.sqrt(1 - alpha_bar)  # [B,C,H,W]
    return score
    
# ----------------------------
# 5?Geodesic Utilities
# ----------------------------
def find_nearest_training_point(y, train_images):
    # y: [B,2], train_images: [N,2]
    B = y.shape[0]
    nearest = []
    for i in range(B):
        dists = ((train_images - y[i:i+1])**2).sum(dim=1)
        idx = dists.argmin()
        nearest.append(train_images[idx])
    return torch.stack(nearest, dim=0)
    
@torch.no_grad()
def linear_path_integral_fast(y, y_target, model, scheduler, t_idx, n_steps=16):
    # y, y_target: [B, D]
    B, D = y.shape
    phi = torch.zeros(B, device=y.device)

    direction = y_target - y
    ds = 1.0 / (n_steps - 1)

    for i in range(n_steps):
        s = i * ds
        y_s = y + s * direction              # [B,D]
        score = score_fn(y_s, model, scheduler, t_idx)  # [B,D]

        # integrand = <score, dy/ds>
        inner = (score * direction).sum(dim=1)

        phi += inner * ds

    return phi

# ------------------------
# Exponential map with new metric
# ------------------------
def exp_map(y, v, model, scheduler, t_idx, train_images=None,
                      lam=1000.0, beta=1.0, n_substeps=8, correct_every=1):
    """
    Exponential map along tangent vector `v` with score correction using the diffusion model.

    Parameters:
    - y: [B,C,H,W] starting point
    - v: [B,C,H,W] tangent vector
    - model: trained UNetSD
    - scheduler: VPScheduler
    - t_idx: diffusion timestep
    - train_images: optional dataset for nearest neighbor
    - lam, beta: Riemannian metric parameters
    - n_substeps: number of substeps
    - correct_every: how often to apply diffusion-based correction
    """
    y_current = y
    dx = v / n_substeps

    for step in range(n_substeps):
        # Compute score
        s = score_fn(y_current, model, scheduler, t_idx)  # [B,C,H,W]

        if beta != 0:
            # Optional: nearest training point
            if train_images is not None:
                y_target = find_nearest_training_point(y_current, train_images)
            else:
                y_target = torch.zeros_like(y_current)
    
            # Path integral factor
            Phi_val = linear_path_integral_fast(y_current, y_target, model, scheduler, t_idx)  # [B]
            #print(Phi_val)
            Phi_val = torch.clamp(Phi_val, -1, 1)
            C = torch.exp(beta * Phi_val).view(-1,1)  # broadcast
        else:
            C = 1

        # Riemannian metric step
        s_norm2 = (s * s).view(s.shape[0], -1).sum(dim=1, keepdim=True)  # [B,1]
        proj = (s * dx).view(s.shape[0], -1).sum(dim=1, keepdim=True)    # [B,1]
        denom = 1 + lam * s_norm2
        ginv_dx = dx - lam * (proj / denom).view(-1,1) * s# * (dx.norm() / (s * s).norm())
        y_current = y_current + C * ginv_dx

        """# Diffusion-based correction (optional)
        if (step + 1) % correct_every == 0:
            t_tensor = torch.tensor([t_idx], device=y_current.device)
            eps = model(y_current, t_tensor)
            alpha_bar = scheduler.alphas_cumprod[t_idx]
            x0_pred = (y_current - eps * torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha_bar)
            y_current = torch.sqrt(alpha_bar) * x0_pred + eps * torch.sqrt(1 - alpha_bar)"""
    """if torch.norm(v) > 0.1:
        displacement = y_current - y
        scale = torch.norm(v) / torch.norm(displacement)
        y_current = y + scale * displacement"""
    return y_current
    
# ------------------------
# Log map shooting with new metric
# ------------------------
"""def log_map_shooting(y, y_target, model, scheduler, t_idx,
                     lam=1000.0, beta=1.0, max_iters=1000, lr=1.0,
                     substep_schedule=(1, 2, 4, 8), train_images=None):
    y = y.detach()
    y_target = y_target.detach()

    v = (y_target - y).detach().clone()
    v.requires_grad_(True)
    optimizer = torch.optim.Adam([v], lr=lr)

    for n_substeps in substep_schedule:
        print(f"=== Using n_substeps = {n_substeps} ===")
        for i in range(max_iters // len(substep_schedule)):
            y_pred = exp_map(
                y, v, model, scheduler, t_idx,
                train_images=train_images,
                lam=lam, beta=beta, n_substeps=n_substeps
            )

            loss = torch.norm(y_pred - y_target)**2

            optimizer.zero_grad()
            loss.backward()

            # gradient clipping to avoid explosions
            torch.nn.utils.clip_grad_norm_([v], max_norm=0.1)

            optimizer.step()

            if i % 50 == 0 or loss.item() < 1e-4:
                print(f"iter {i}, loss {loss.item():.6f}")

            if loss.item() < 1e-4:
                break

    return v.detach()"""
    
def log_map_shooting(y, y_target, model, scheduler, t_idx,
                     lam=1000.0, beta=1.0, max_iters=1000, lr=0.1,
                     n_substeps_schedule=[1, 2, 4, 8, 16], train_images=None):
    y = y.detach()
    y_target = y_target.detach()
    tol = 1e-4
    momentum_gamma = 0.9
    
    v = (y_target - y).detach().clone()
    v.requires_grad_(True)
    
    momentum = torch.zeros_like(v)
    best_v = v.clone()
    best_loss = float('inf')

    for idx, n_substeps in enumerate(n_substeps_schedule):
        n_iters = max_iters // len(n_substeps_schedule)
        V = best_v
        for i in range(n_iters):
            y_pred = exp_map(y, v, model, scheduler, t_idx,
                             lam=lam, beta=beta, n_substeps=n_substeps,
                             train_images=train_images)
            residual = y_target - y_pred

            loss = residual.norm()**2
            print(f"n_substeps={n_substeps}, iter={i}, loss={loss.item():.6f}")

            if loss.item() < tol:
                break

            # normalize residual to prevent huge jumps
            step = lr * residual / (residual.norm() + 1e-8)

            # apply momentum
            momentum = momentum_gamma * momentum + step
            v = v + momentum

            # track best v only at the **max substeps stage**
            if n_substeps == n_substeps_schedule[-1] and loss.item() < best_loss:
                best_loss = loss.item()
                best_v = v.clone()

    return best_v

# ----------------------------
# 6?Training Function
# ----------------------------
def train_diffusion_2d(model, dataset, scheduler, epochs=500, batch_size=64, lr=1e-3):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    for epoch in range(epochs):
        for x0 in loader:
            x0 = x0.to(device)
            t = torch.randint(0, scheduler.num_timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            x_t = scheduler.q_sample(x0, t, noise)
            eps_pred = model(x_t, t.float())
            loss = ((noise - eps_pred)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ----------------------------
# 7?Main Script
# ----------------------------
device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Dataset & Model
# ------------------------
dataset = Curve2DDataset(n_points=500)
#model = MLPDiffusion2D().to(device)
scheduler = VPScheduler(num_timesteps=1000)

# ------------------------
# Train diffusion model
# ------------------------
#print("Training diffusion model...")
#train_diffusion_2d(model, dataset, scheduler, epochs=100000)

# Save the model state
model_path = "mlp_diffusion_2d.pth"
#torch.save(model.state_dict(), model_path)
#print(f"Model saved to {model_path}")

# Create the same model architecture
model = MLPDiffusion2D().to(device)

# Load the saved weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # set to evaluation mode if not training

# ------------------------
# Select two random points
# ------------------------
x = dataset.data[0:1].to(device)
xA_ = torch.randn_like(x)
xB_ = torch.randn_like(x)
num_steps = 1000

xA = ddim_sample(model, scheduler, xA_, torch.linspace(scheduler.num_timesteps-1, 0, num_steps, dtype=torch.long, device=device))
xB = ddim_sample(model, scheduler, xB_, torch.linspace(scheduler.num_timesteps-1, 0, num_steps, dtype=torch.long, device=device))

# ------------------------
# Beta values to compare
# ------------------------
beta_values = [-0.5, -0.375, -0.25, -0.125, 0]
colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
n_steps = 20
lam = 1.0
t_idx = 30

plt.figure(figsize=(6,6))
plt.scatter(dataset.data[:,0], dataset.data[:,1], alpha=0.2, label="Dataset")

# ------------------------
# Compute geodesics for each beta
# ------------------------
beta_stats = []   # store (beta, avg_dist, max_dist, min_dist)

for beta, color in zip(beta_values, colors):
    print(f"\n=== Beta = {beta} ===")

    # Log map shooting
    v = 1.5 * log_map_shooting(
        xA, xB, model, scheduler, t_idx=t_idx,
        lam=lam, beta=beta
    )
    
    # Sample points along geodesic
    geodesic_points = []
    for i in range(n_steps):
        s = i / (n_steps - 1)
        y_s = exp_map(xA, s*v, model, scheduler,
                      t_idx=t_idx, beta=beta, n_substeps=16)
        x0_pred = ddim_sample(
            model, scheduler, y_s,
            torch.linspace(29, 0, 30, dtype=torch.long, device=device)
        )
        geodesic_points.append(x0_pred.detach().cpu())

    geodesic_points = torch.cat(geodesic_points, dim=0)        # [K,2]

    # -----------------------------------------------
    # Compute distances to manifold
    # -----------------------------------------------
    data = dataset.data.to(geodesic_points.device)

    dists = []
    for p in geodesic_points:
        diff = data - p
        dist = torch.sqrt((diff*diff).sum(dim=1))
        dists.append(dist.min().item())

    avg_dist = float(np.mean(dists))
    max_dist = float(np.max(dists))
    min_dist = float(np.min(dists))

    # Store stats but do NOT print yet
    beta_stats.append((beta, avg_dist, max_dist, min_dist))

    # Plot
    pts = geodesic_points.numpy()
    plt.plot(pts[:,0], pts[:,1], f'{color}-o', label=f'Beta={beta}')

# ------------------------
# Print clean summary AFTER ALL GEODESICS
# ------------------------
print("\n==================== Manifold Distance Summary ====================")
for beta, avg_dist, max_dist, min_dist in beta_stats:
    print(f"Beta={beta:>3} | avg={avg_dist:.4f} | max={max_dist:.4f} | min={min_dist:.4f}")
print("===================================================================\n")

    
# ------------------------
# Linear interpolation for reference
# ------------------------
xA_ = xA_.cpu()
xB_ = xB_.cpu()
device = next(model.parameters()).device

linear_points = torch.stack([
    ddim_sample(model, scheduler, (xA_* (1-s) + xB_*s).to(device),
                torch.linspace(scheduler.num_timesteps-1, 0, num_steps, dtype=torch.long, device=device))
    for s in torch.linspace(0,1,n_steps)
], dim=0).squeeze().cpu().detach().numpy()

plt.plot(linear_points[:,0], linear_points[:,1], 'k--', label="Linear interpolation")

# ------------------------
# Endpoints
# ------------------------
plt.scatter([xA[0,0].item(), xB[0,0].item()],
            [xA[0,1].item(), xB[0,1].item()],
            c='k', label="Endpoints")


plt.xlim(-1.32,1.32)
plt.ylim(-1.32,1.32)
plt.axis('equal')
plt.legend()
plt.title("Geodesics on 2D Curve for Different Beta Values")
plt.savefig("Test.png")
#plt.show()
