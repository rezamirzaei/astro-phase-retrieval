# %% [markdown]
# # Crystallographic Phase Retrieval
#
# This notebook demonstrates **phase retrieval for X-ray crystallography** using the same
# iterative projection algorithms used for astronomical wavefront sensing.
#
# ## The Crystallographic Phase Problem
#
# When X-rays scatter off a crystal lattice, the detector records **diffraction intensities**
# $|F(\mathbf{h})|^2$ but loses the **phase** $\varphi(\mathbf{h})$. Recovering these phases
# from measured intensities alone is the *crystallographic phase problem* — one of the
# foundational challenges in structural biology and materials science.
#
# We demonstrate the complete pipeline:
# 1. **Download** a real crystal structure from the Crystallography Open Database (COD)
# 2. **Parse** the CIF file and extract unit-cell parameters + atom sites
# 3. **Simulate** a 2-D diffraction pattern from the structure factors
# 4. **Run** iterative phase retrieval algorithms (ER, HIO, RAAR, WF)
# 5. **Visualise** recovered phases, electron density, and convergence
# 6. **Compare** algorithm performance using the crystallographic R-factor

# %%
import matplotlib

matplotlib.use('Agg')
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# Phase retrieval modules
from src.data.crystallography import (  # noqa: E402
    available_cod_presets,
    run_crystallography_retrieval,
    simulate_diffraction,
)
from src.models.crystallography import AtomSite, CrystalStructure  # noqa: E402
from src.visualization.crystallography_plots import (  # noqa: E402
    plot_crystal_summary,
    plot_crystallography_comparison,
    plot_crystallography_result,
    plot_diffraction_pattern,
    plot_electron_density,
    plot_r_factor_comparison,
)
from src.visualization.plots import save_figure  # noqa: E402

OUTPUT_DIR = Path('notebooks/outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRID_SIZE = 128

print('Available COD presets:')
for key, desc in available_cod_presets().items():
    print(f'  {key:15s} — {desc}')

# %% [markdown]
# ## 1. Crystal Structure: NaCl (Rock Salt)
#
# We start with Sodium Chloride — one of the most studied crystal structures.
# The face-centred cubic (FCC) lattice with space group **Fm-3m** has Na at the
# origin and Cl at (½, ½, ½).

# %%
# Define NaCl structure directly (no network needed)
nacl = CrystalStructure(
    cod_id='1000041',
    formula='NaCl',
    space_group='F m -3 m',
    a=5.6400, b=5.6400, c=5.6400,
    alpha=90.0, beta=90.0, gamma=90.0,
    atoms=[
        AtomSite(label='Na1', symbol='Na', x=0.0, y=0.0, z=0.0),
        AtomSite(label='Cl1', symbol='Cl', x=0.5, y=0.5, z=0.5),
    ],
)

print(f'Crystal: {nacl.formula}')
print(f'Space group: {nacl.space_group}')
print(f'Unit cell: a={nacl.a:.4f} Å, b={nacl.b:.4f} Å, c={nacl.c:.4f} Å')
print(f'Angles: α={nacl.alpha}°, β={nacl.beta}°, γ={nacl.gamma}°')
print(f'Atoms: {len(nacl.atoms)}')
for atom in nacl.atoms:
    print(f'  {atom.label}: {atom.symbol} at ({atom.x:.4f}, {atom.y:.4f}, {atom.z:.4f})')

# %% [markdown]
# ## 2. Simulate Diffraction Pattern
#
# We compute the 2-D structure factors $F(h,k)$ for the $l=0$ reciprocal-space
# plane, then plot the diffraction intensities $|F(h,k)|^2$ with a thermal
# (Debye-Waller) damping factor.

# %%
# Simulate diffraction
pattern = simulate_diffraction(nacl, grid_size=GRID_SIZE)

print(f'Diffraction pattern shape: {pattern.image.shape}')
print(f'Total intensity: {pattern.image.sum():.6f}')
print(f'Max intensity: {pattern.image.max():.6e}')
print(f'Space group: {pattern.space_group}')

# Plot
fig = plot_diffraction_pattern(pattern)
save_figure(fig, OUTPUT_DIR / 'cryst_diffraction_nacl.png')
plt.close(fig)
print('Saved: cryst_diffraction_nacl.png')

# %% [markdown]
# ## 3. Phase Retrieval with HIO
#
# We run the **Hybrid Input-Output (HIO)** algorithm on the simulated diffraction
# pattern to recover the lost phases. The algorithm iterates between the
# reciprocal-space constraint (measured amplitudes) and a real-space support
# constraint.

# %%
# Run HIO phase retrieval
result_hio = run_crystallography_retrieval(
    pattern,
    algorithm_name='hio',
    max_iterations=200,
    beta=0.9,
    random_seed=42,
)

print(f'Algorithm: {result_hio.algorithm.value.upper()}')
print(f'Iterations: {result_hio.n_iterations}')
print(f'Converged: {result_hio.converged}')
print(f'R-factor: {result_hio.r_factor:.4f}')
print(f'Time: {result_hio.elapsed_seconds:.2f}s')

# Full result plot
fig = plot_crystallography_result(pattern, result_hio)
save_figure(fig, OUTPUT_DIR / 'cryst_result_hio.png')
plt.close(fig)

# Summary
fig = plot_crystal_summary(pattern, result_hio)
save_figure(fig, OUTPUT_DIR / 'cryst_summary_hio.png')
plt.close(fig)

print('Saved: cryst_result_hio.png, cryst_summary_hio.png')

# %% [markdown]
# ## 4. Algorithm Comparison
#
# We compare four phase-retrieval algorithms on the same NaCl diffraction data:
# - **ER** (Error Reduction) — monotone convergence, slow
# - **HIO** (Hybrid Input-Output) — fast escape from local minima
# - **RAAR** (Relaxed Averaged Alternating Reflections) — robust convergence
# - **WF** (Wirtinger Flow) — gradient-based, provably optimal

# %%
algorithms = ['er', 'hio', 'raar', 'wf']
results = {}

for alg in algorithms:
    res = run_crystallography_retrieval(
        pattern,
        algorithm_name=alg,
        max_iterations=200,
        random_seed=42,
    )
    results[alg.upper()] = res
    print(
        f'{alg.upper():>5s}  R={res.r_factor:.4f}  '
        f'iter={res.n_iterations:4d}  time={res.elapsed_seconds:.2f}s'
    )

# R-factor comparison bar chart
fig = plot_r_factor_comparison(results)
save_figure(fig, OUTPUT_DIR / 'cryst_r_factor_comparison.png')
plt.close(fig)

# Multi-algorithm comparison grid
fig = plot_crystallography_comparison(pattern, results)
save_figure(fig, OUTPUT_DIR / 'cryst_algorithm_comparison.png')
plt.close(fig)

print('\nSaved: cryst_r_factor_comparison.png, cryst_algorithm_comparison.png')

# %% [markdown]
# ## 5. Electron Density Maps
#
# The recovered phases combined with the measured amplitudes allow us to compute
# the **electron density** via inverse Fourier transform:
#
# $$\rho(\mathbf{r}) = \sum_{\mathbf{h}} |F(\mathbf{h})|
#   \exp[i\varphi(\mathbf{h})] \exp(-2\pi i\, \mathbf{h}\cdot\mathbf{r})$$
#
# The electron density peaks reveal atom positions in the unit cell.

# %%
# Best result
best_alg = min(results, key=lambda k: results[k].r_factor)
best = results[best_alg]
print(f'Best algorithm: {best_alg} (R = {best.r_factor:.4f})')

fig = plot_electron_density(best)
save_figure(fig, OUTPUT_DIR / 'cryst_electron_density.png')
plt.close(fig)
print('Saved: cryst_electron_density.png')

# %% [markdown]
# ## 6. More Crystal Structures
#
# Let's apply the same pipeline to additional crystal structures to demonstrate
# versatility.

# %%
# Diamond structure (cubic, space group Fd-3m)
diamond = CrystalStructure(
    cod_id='1010927',
    formula='C',
    space_group='F d -3 m',
    a=3.5670, b=3.5670, c=3.5670,
    atoms=[
        AtomSite(label='C1', symbol='C', x=0.0, y=0.0, z=0.0),
        AtomSite(label='C2', symbol='C', x=0.25, y=0.25, z=0.25),
    ],
)

# Quartz (trigonal, SiO2)
quartz = CrystalStructure(
    cod_id='1011097',
    formula='SiO2',
    space_group='P 32 2 1',
    a=4.9134, b=4.9134, c=5.4052,
    alpha=90.0, beta=90.0, gamma=120.0,
    atoms=[
        AtomSite(label='Si1', symbol='Si', x=0.4697, y=0.0, z=0.0),
        AtomSite(label='O1', symbol='O', x=0.4135, y=0.2669, z=0.1191),
    ],
)

structures = {'NaCl': nacl, 'Diamond': diamond, 'Quartz': quartz}

print(f'{"Crystal":>10s}  {"R-factor":>10s}  {"Iter":>6s}  {"Time (s)":>10s}')
print('-' * 45)

for name, crystal in structures.items():
    pat = simulate_diffraction(crystal, grid_size=GRID_SIZE)
    res = run_crystallography_retrieval(
        pat, algorithm_name='raar', max_iterations=200, random_seed=42
    )
    print(f'{name:>10s}  {res.r_factor:10.4f}  {res.n_iterations:6d}  {res.elapsed_seconds:10.2f}')

# %% [markdown]
# ## Summary
#
# This notebook demonstrated that the **same iterative phase-retrieval algorithms**
# used for astronomical wavefront sensing can be directly applied to
# **X-ray crystallographic data**:
#
# - The diffraction intensity $|F(h,k)|^2$ plays the role of the PSF intensity
# - The crystallographic R-factor replaces the Strehl ratio as quality metric
# - Recovered phases → inverse FFT → electron density map
# - HIO and RAAR typically give the best results, matching known behaviour
#
# For real-world data, use the COD download functions or the web UI to fetch
# CIF files from the Crystallography Open Database.


