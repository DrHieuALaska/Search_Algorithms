"""
DE Convergence GIF — v4
- Info box (Current Best / Global Best) in top-right corner of each subplot (2D overlay)
- Population moves SMOOTHLY toward the true minimum (interpolated trajectory)
- No floating text on the 3D points
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.mplot3d import Axes3D  
from PIL import Image
import io, os, math
from matplotlib.gridspec import GridSpec


# functions (10D) 
DIMENSION = 10

def sphere_nd(X):
    return np.sum(X**2)

def rastrigin_nd(X):
    A = 10
    return A * len(X) + np.sum(X**2 - A * np.cos(2 * np.pi * X))

def rosenbrock_nd(X):
    return np.sum(100*(X[1:] - X[:-1]**2)**2 + (X[:-1] - 1)**2)

def make_slice_fn(nd_func, dim=DIMENSION, fixed_val=0.0):
    def fn2d(x1, x2):
        X = np.full(dim, fixed_val)
        X[0] = x1; X[1] = x2
        return nd_func(X)
    return np.vectorize(fn2d)

FUNC_META = {
    'sphere': {
        'nd_func':   sphere_nd,
        'fn_label':  'Sphere',
        'bounds':    (-5.12, 5.12),
        'fixed_val': 0.0,
        'pop_size':  200,
        'F': 0.18, 'CR': 0.8,
        'f_target':  1e-6,
        'optimum':   np.zeros(DIMENSION),
        'z_clip':    55,
    },
    'rastrigin': {
        'nd_func':   rastrigin_nd,
        'fn_label':  'Rastrigin',
        'bounds':    (-5.12, 5.12),
        'fixed_val': 0.0,
        'pop_size':  55,
        'F': 0.14, 'CR': 0.5,
        'f_target':  1.0,
        'optimum':   np.zeros(DIMENSION),
        'z_clip':    90,
    },
    'rosenbrock': {
        'nd_func':   rosenbrock_nd,
        'fn_label':  'Rosenbrock',
        'bounds':    (-1.5, 1.75),    # tighter display bounds → flatter, prettier surface like reference
        'fixed_val': 0.0,
        'pop_size':  200,
        'F': 0.45, 'CR': 0.95,
        'f_target':  1.0,
        'optimum':   np.ones(DIMENSION),
        'z_clip':    3500,          # within (-2,2) surface peaks ~3600, clip for clean view
    },
}

FUNCS = ['sphere', 'rastrigin', 'rosenbrock']

# Surface
def build_surface(meta, n_grid=70):
    lo, hi = meta['bounds']
    x = np.linspace(lo, hi, n_grid)
    y = np.linspace(lo, hi, n_grid)
    X, Y = np.meshgrid(x, y)
    fn2d = make_slice_fn(meta['nd_func'], fixed_val=meta['fixed_val'])
    Z = fn2d(X, Y)
    Z = np.clip(Z, 0, meta['z_clip'])
    return X, Y, Z

# Smooth population trajectories
def build_trajectories_smooth(meta, fitness_snapshots, n_pop=30, rng=None):
    """
    Each individual has its OWN smooth path from a random start → optimum.
    Speed is governed by the global fitness decay curve so the cloud
    visibly rushes in during the steep drop and slows near convergence.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    nd_func = meta['nd_func']
    opt2    = meta['optimum'][:2]
    lo, hi  = meta['bounds']
    f0      = fitness_snapshots[0]
    n_snaps = len(fitness_snapshots)

    # Individual start positions — spread uniformly across search space
    starts = rng.uniform(lo, hi, (n_pop, 2))

    # Convergence progress per snapshot: 0 (start) → 1 (fully converged)
    # Use a power-law mapping of the normalised fitness so movement is visible
    f_norm = fitness_snapshots / (f0 + 1e-300)          # 1 → ~0
    # progress = 1 - f_norm^alpha  (alpha controls speed of approach)
    alpha    = 0.30
    progress = 1.0 - np.clip(f_norm, 0, 1) ** alpha     # 0 → 1

    pop_hist  = np.empty((n_snaps, n_pop, 2))
    best_hist = np.empty((n_snaps, 2))

    for ti, p in enumerate(progress):
        # Each individual interpolates linearly start → optimum
        # Add a shrinking noise so they don't all stack exactly at opt
        noise_scale = max(1.0 - p, 1e-4) * (hi - lo) * 0.18
        positions   = (1 - p) * starts + p * opt2
        positions  += rng.normal(0, noise_scale, (n_pop, 2))
        positions   = np.clip(positions, lo, hi)
        pop_hist[ti] = positions

        # Best = individual closest to optimum at this snapshot
        dists = np.linalg.norm(positions - opt2, axis=1)
        best_hist[ti] = positions[np.argmin(dists)]

    return pop_hist, best_hist


# PIL helper
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110,
                bbox_inches='tight', facecolor='white')
    buf.seek(0)
    return Image.open(buf).copy()


# ── Draw one frame ────────────────────────────────────────────────────────────
def draw_frame(fig, axes, infoboxes, surfaces, pop_hists, best_hists,
               gbest_list, frame_fitness_all, global_best_all,
               frame_eval, fi, n_frames, func_name):

    for ax in axes:
        for coll in ax.collections[1:]:
            coll.remove()
        for txt in ax.texts:
            txt.remove()

    for col_idx, fname in enumerate(FUNCS):
        ax      = axes[col_idx]
        ibox    = infoboxes[col_idx]   
        meta    = FUNC_META[fname]
        lo, hi  = meta['bounds']
        fn2d    = make_slice_fn(meta['nd_func'], fixed_val=meta['fixed_val'])

        pop_xy   = pop_hists[fname][fi]
        best_xy  = best_hists[fname][fi]
        gbest_xy = gbest_list[fname]
        cur_f    = frame_fitness_all[fname][fi]
        glob_f   = global_best_all[fname]

        # z values
        pz  = np.array([float(np.clip(fn2d(x, y), 0, meta['z_clip'])) for x, y in pop_xy])
        bz  = float(np.clip(fn2d(*best_xy),  0, meta['z_clip']))
        gz  = float(np.clip(fn2d(*gbest_xy), 0, meta['z_clip']))

        # population
        norm_pz = Normalize(vmin=0, vmax=meta['z_clip'])
        ax.scatter(pop_xy[:,0], pop_xy[:,1], pz,
                   c="#ff00d9", s=32, alpha=0.90,
                   edgecolors="#000000", linewidths=0.5, zorder=5)

        # current best
        ax.scatter([best_xy[0]], [best_xy[1]], [bz],
                   c='#ff9900', s=120, alpha=1.0,
                   edgecolors='#880000', linewidths=1.2,
                   marker='*', zorder=10)

        # global best
        ax.scatter([gbest_xy[0]], [gbest_xy[1]], [gz],
                   c='#dd0000', s=130, alpha=1.0,
                   edgecolors='#440000', linewidths=1.5,
                   marker='D', zorder=11)

        # info box in bottem right
        ibox.clear()
        ibox.set_xlim(0, 1); ibox.set_ylim(0, 1)
        ibox.axis('off')
        def fmt(v):
            return f"{v:.4f}" if abs(v) >= 1e-4 else f"{v:.4e}"
        txt = (f"Global Best:  {fmt(glob_f)}\n"
               f"Current Best: {fmt(cur_f)}")
        ibox.text(0.97, 0.80, txt,
                  transform=ibox.transAxes,
                  fontsize=8.5, family='monospace',
                  verticalalignment='top', horizontalalignment='right',
                  color='#111111',
                  bbox=dict(boxstyle='round,pad=0.5',
                            facecolor='#fffff0',
                            edgecolor='#aaaaaa',
                            alpha=0.92,
                            linewidth=0.8))

    # suptile
    fig.suptitle(
        f'Algorithm: {func_name}          Func Evals: {frame_eval:,}',
        fontsize=13, fontweight='bold', color='#222222', y=0.99
    )


def make_gif(df, func_name, out_path, n_frames=90, n_pop=10):
    print("Loading data & building surfaces …")
    f_cols = [c for c in df.columns if c.startswith('f') and c[1:].isdigit()]
    evals  = np.array([int(c[1:]) for c in f_cols])

    early = np.linspace(0, len(f_cols) // 3,     n_frames * 2 // 3, dtype=int)
    late  = np.linspace(len(f_cols) // 3, len(f_cols) - 1,
                        n_frames - n_frames * 2 // 3, dtype=int)
    fidx       = np.unique(np.concatenate([early, late]))
    n_frames   = len(fidx)
    frame_evals = evals[fidx]

    surfaces          = {}
    pop_hists         = {}
    best_hists        = {}
    gbest_list        = {}
    frame_fitness_all = {}
    global_best_all   = {}

    for fname in FUNCS:
        meta = FUNC_META[fname]
        sub  = df[df['func_name'] == fname]
        med  = sub[f_cols].median(axis=0).values.astype(float)
        frame_fitness_all[fname] = med[fidx]
        global_best_all[fname]   = float(sub['best_fitness'].min())
        surfaces[fname]          = build_surface(meta, n_grid=70)
        rng = np.random.default_rng(99)
        pop_hists[fname], best_hists[fname] = build_trajectories_smooth(
            meta, frame_fitness_all[fname], n_pop=n_pop, rng=rng
        )
        gbest_list[fname] = best_hists[fname][-1].copy()


    fig = plt.figure(figsize=(17, 7.2), facecolor='white')
    fig.patch.set_facecolor('white')

    

    gs = GridSpec(2, 3, figure=fig,
                  height_ratios=[6, 1],
                  hspace=0.02, wspace=0.08,
                  top=0.93, bottom=0.02, left=0.02, right=0.98)

    axes     = []
    infoboxes = []

    for col_idx, fname in enumerate(FUNCS):
        ax   = fig.add_subplot(gs[0, col_idx], projection='3d')
        ibox = fig.add_subplot(gs[1, col_idx])
        axes.append(ax)
        infoboxes.append(ibox)

        meta   = FUNC_META[fname]
        lo, hi = meta['bounds']
        SX, SY, SZ = surfaces[fname]

        # Use LogNorm for Rosenbrock (huge z range) so color variation is visible
        if fname == 'rosenbrock':
            surf_norm = LogNorm(vmin=max(SZ.min(), 1e-1), vmax=meta['z_clip'])
        else:
            surf_norm = Normalize(vmin=0, vmax=meta['z_clip'])

        ax.plot_surface(
            SX, SY, SZ,
            cmap='jet', alpha=0.75,
            rstride=1, cstride=1,
            linewidth=0, antialiased=True, shade=True,
            norm=surf_norm
        )
        ax.set_xlabel('x₁', fontsize=9, color='#333333', labelpad=4)
        ax.set_ylabel('x₂', fontsize=9, color='#333333', labelpad=4)
        ax.set_zlabel('f(x)', fontsize=9, color='#333333', labelpad=4)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_zlim(0, meta['z_clip'])
        ax.set_title(meta['fn_label'], fontsize=12,
                     fontweight='bold', color='#222222', pad=6)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#cccccc')
        ax.yaxis.pane.set_edgecolor('#cccccc')
        ax.zaxis.pane.set_edgecolor('#cccccc')
        ax.grid(color='#cccccc', linestyle='-', linewidth=0.4, alpha=0.7)
        ax.tick_params(labelsize=7, colors='#444444')
        ax.view_init(elev=22, azim=-55)

        ibox.set_facecolor('white')
        ibox.axis('off')

    # render frames
    print(f"Rendering {n_frames} frames …")
    frames_pil = []
    azim_start, azim_range = -60, 100

    for fi in range(n_frames):
        t    = fi / max(n_frames - 1, 1)
        azim = azim_start + azim_range * t
        elev = 22 + 7 * math.sin(math.pi * t)
        for ax in axes:
            ax.view_init(elev=elev, azim=azim)

        draw_frame(fig, axes, infoboxes, surfaces,
                   pop_hists, best_hists, gbest_list,
                   frame_fitness_all, global_best_all,
                   int(frame_evals[fi]), fi, n_frames, func_name)

        frames_pil.append(fig_to_pil(fig))
        if (fi + 1) % 20 == 0:
            print(f"  {fi+1}/{n_frames}")

    plt.close(fig)

    # save
    dur = [110] * n_frames
    dur[0] = 900; dur[-1] = 1800
    frames_pil[0].save(
        out_path, save_all=True,
        append_images=frames_pil[1:],
        duration=dur, loop=0, optimize=True
    )
    print(f"\n {out_path}  ({os.path.getsize(out_path)//1024} KB,  {n_frames} frames)")


if __name__ == '__main__':
    func_name = ["abc", "cuckoo_search", "de", "firefly", "pso", "ga", "tlbo"]
    for i in range(len(func_name)):
        print(f"Function's name: {func_name[i]}" )
        df = pd.read_csv(f'../Data_with_function_evaluations/trials_data/trials_data_10D/{func_name[i]}_results.csv')
        os.makedirs('3D_Gif', exist_ok=True)
        make_gif(df, func_name[i], f'3D_Gif/{func_name[i]}.gif', n_frames=90, n_pop=10)
        print("Done!")