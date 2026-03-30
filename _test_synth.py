"""Test RAAR on a known synthetic problem to validate the algorithm."""
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

n = 128
y, x = np.mgrid[-1:1:n*1j, -1:1:n*1j]
r = np.sqrt(x**2 + y**2)
pupil_amp = np.where(r < 0.9, 1.0, 0.0).astype(np.float64)
support = pupil_amp > 0

# True phase: astigmatism-like
true_phase = 0.5 * (x**2 - y**2) + 0.3 * x * y
true_phase[~support] = 0.0

# Observed PSF (noiseless)
g_true = pupil_amp * np.exp(1j * true_phase)
G_true = fftshift(fft2(ifftshift(g_true)))
obs_psf = np.abs(G_true)**2
obs_psf /= obs_psf.sum()
target_amp = np.sqrt(obs_psf)

# --------------- RAAR ---------------
beta = 0.9
rng = np.random.default_rng(42)
phase0 = rng.uniform(-0.3, 0.3, size=(n, n))
g = pupil_amp * np.exp(1j * phase0)
costs_raar = []
for i in range(300):
    G = fftshift(fft2(ifftshift(g)))
    modelled = np.abs(G)
    scale = target_amp.sum() / max(modelled.sum(), 1e-30)
    costs_raar.append(float(np.sum((target_amp - modelled * scale)**2)))

    G_proj = target_amp * np.exp(1j * np.angle(G))
    p_f_g = fftshift(ifft2(ifftshift(G_proj)))     # P_F(g)
    r_f_g = 2.0 * p_f_g - g                         # R_F(g)

    # P_S(R_F(g))
    p_s_rf = np.zeros_like(r_f_g)
    p_s_rf[support] = pupil_amp[support] * np.exp(1j * np.angle(r_f_g[support]))

    # R_S(R_F(g)) = 2·P_S(R_F(g)) − R_F(g)
    r_s_rf = 2.0 * p_s_rf - r_f_g

    # RAAR: g_new = β/2·(R_S·R_F + I)·g + (1−β)·P_F·g
    g = (beta / 2.0) * (r_s_rf + g) + (1.0 - beta) * p_f_g

final_phase_raar = np.angle(g)
final_phase_raar[~support] = 0
fp = final_phase_raar[support] - final_phase_raar[support].mean()
tp = true_phase[support] - true_phase[support].mean()
print("=== RAAR (synthetic) ===")
print(f"  cost[0]={costs_raar[0]:.6f}  cost[-1]={costs_raar[-1]:.6f}")
print(f"  cost decreasing: {costs_raar[-1] < costs_raar[0]}")
print(f"  residual vs truth: {np.sqrt(np.mean((fp - tp)**2)):.4f} rad")
print(f"  correlation w/ truth: {np.corrcoef(fp, tp)[0, 1]:.4f}")

# --------------- ER ---------------
rng2 = np.random.default_rng(42)
phase0 = rng2.uniform(-0.3, 0.3, size=(n, n))
g = pupil_amp * np.exp(1j * phase0)
costs_er = []
for i in range(300):
    G = fftshift(fft2(ifftshift(g)))
    modelled = np.abs(G)
    scale = target_amp.sum() / max(modelled.sum(), 1e-30)
    costs_er.append(float(np.sum((target_amp - modelled * scale)**2)))
    G_prime = target_amp * np.exp(1j * np.angle(G))
    g_prime = fftshift(ifft2(ifftshift(G_prime)))
    g_new = np.zeros_like(g_prime)
    g_new[support] = pupil_amp[support] * np.exp(1j * np.angle(g_prime[support]))
    g = g_new

final_phase_er = np.angle(g)
final_phase_er[~support] = 0
fp = final_phase_er[support] - final_phase_er[support].mean()
print("\n=== ER (synthetic) ===")
print(f"  cost[0]={costs_er[0]:.6f}  cost[-1]={costs_er[-1]:.6f}")
print(f"  cost decreasing: {costs_er[-1] < costs_er[0]}")
print(f"  residual vs truth: {np.sqrt(np.mean((fp - tp)**2)):.4f} rad")
print(f"  correlation w/ truth: {np.corrcoef(fp, tp)[0, 1]:.4f}")

# Also check: is the cost metric itself sensible?
# Perfect reconstruction cost
G_perfect = fftshift(fft2(ifftshift(g_true)))
modelled_perfect = np.abs(G_perfect)
scale_p = target_amp.sum() / max(modelled_perfect.sum(), 1e-30)
perfect_cost = float(np.sum((target_amp - modelled_perfect * scale_p)**2))
print(f"\n=== Perfect reconstruction cost: {perfect_cost:.10f} ===")


