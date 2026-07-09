"""
final_figure_v4.py
==================
4-panel (2x2) figure for the PySR NTM bias results.

  (a) top-left  : Complexity vs bootstrap NMSE for ALL best equations,
                   both phi_tau (flux) and Delta_tau (W1) on one plot.
                   The 1.5x-rule winner for each output is highlighted.

  (b) top-right : The two selected best-of-best equations rendered as
                   LaTeX math text, stacked vertically.

  (c) bottom-left: Variable occurrence bar charts — how many of the 11
                    bias-run equations contain each NTM parameter, shown
                    side by side for phi_tau and Delta_tau.

  (d) bottom-right: Residual scatter plots (y_true - y_pred vs y_true)
                     for the two selected equations on the validation set,
                     shown side by side.

Usage
-----
    python final_figure_v4.py

Output
------
    final_figure_v4.png  saved to the UCM_AD_ML directory.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sympy as sp
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pysr_data_processor_v2 import (
    load_scaling_params,
    apply_scaling,
    extract_terms_and_ops,
    rank_best_equations_by_bootstrap,
    select_best_of_best,
)

# ===========================================================================
# PATHS
# ===========================================================================
_BASE       = r'C:\Users\USER\Documents\Code\UCM_AD_ML\datasets\sim_csv_v11'
TRAIN_DIR   = os.path.join(_BASE, 'train')
E3_DATA_DIR = os.path.join(_BASE, 'e3_data')
VAL_CSV     = os.path.join(E3_DATA_DIR, 'bias_val_e3.csv')

FLUX_TRAIN_DIR = os.path.join(TRAIN_DIR, 'flux_only_bias_train_e3')
W1_TRAIN_DIR   = os.path.join(TRAIN_DIR, 'w1_only_bias_train_e3')

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'final_figure_v4.png')

FEATURE_COLS        = ['gamma1', 'lambda1', 'delta', 'epsilon', 'NRow', 'NCol']
BEST_OF_BEST_FACTOR = 1.5

# Set to an integer to override the W1 (Delta_tau) winner by true PySR complexity
# (from model_equations.csv, with complexity_of_constants=5).
# Set to None to use the automatic best-of-best selection.
W1_COMPLEXITY_OVERRIDE = 25

# Panel (a) display mode — choose one:
#   'combined'      : single plot, both outputs overlaid (blue/red series)
#   'split_best'    : two sub-panels, one per output, best.txt points only
#   'full_pareto'   : two sub-panels, mean Train/Val NMSE across all 11 runs
#   'winner_pareto' : two sub-panels, Train/Val NMSE from the single winning
#                     run's model_equations.csv (the actual Pareto frontier
#                     that produced the chosen best-of-best expression)
PANEL_A_MODE = 'winner_pareto'

# When True, the validation curve in 'full_pareto' mode is recomputed by
# evaluating every equation on bias_val_e3.csv (consistent with panel b).
# When False, NMSE_Validation from metrics_summary.csv is used (PySR's
# internal split — faster but a different dataset than panel b).
USE_HOLDOUT_VAL = False

# ===========================================================================
# AESTHETICS
# ===========================================================================
VAR_LABELS = {
    'x0': r'$\gamma$', 'x1': r'$\lambda$',
    'x2': r'$\delta$',  'x3': r'$\varepsilon$',
    'x4': r'$N_1$',     'x5': r'$N_2$',
}
VAR_COLORS = {
    'x0': '#E8A0A0', 'x1': '#A0B8D8', 'x2': '#A0D8A0',
    'x3': '#C8A0D8', 'x4': '#D8D890', 'x5': '#C8C8C8',
}
ORDERED_VARS = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5']

FLUX_COLOR  = '#4878CF'   # blue  for phi_tau
W1_COLOR    = '#D65F5F'   # red   for Delta_tau
WIN_EDGE    = '#FFD700'   # gold  outline for the selected winner points

plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       16,
    'axes.titlesize':  20,
    'axes.labelsize':  19,   # +25%
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
})


# ===========================================================================
# DATA HELPERS  (shared with final_figure_v3.py logic)
# ===========================================================================
def collect_best_equations(dataset_dir):
    """Walk dataset_dir and parse each best.txt → list of candidate dicts."""
    results = []
    pattern = re.compile(r"Selected Best Function:\s*(\d+),\s*(.*)")
    for root, _, files in os.walk(dataset_dir):
        if 'best.txt' in files:
            with open(os.path.join(root, 'best.txt'), 'r') as fh:
                first_line = fh.readline()
            m = pattern.match(first_line.strip())
            if m:
                results.append({
                    'run':        os.path.basename(root),
                    'run_dir':    root,
                    'complexity': int(m.group(1)),
                    'equation':   m.group(2).strip(),
                })
    print(f"  Collected {len(results)} candidates from {dataset_dir}")
    return results


def compute_predictions(winner, val_df, target_col):
    """Full-set (y_true_scaled, y_pred_scaled) for the winning equation."""

    model_eq_path = os.path.join(winner['run_dir'], 'model_equations.csv')
    scaling       = load_scaling_params(model_eq_path)
    X_sc, y_sc, _ = apply_scaling(val_df, target_col, scaling,
                                   feature_cols=FEATURE_COLS)
    syms  = sp.symbols("x0 x1 x2 x3 x4 x5")
    expr  = sp.sympify(winner['equation'])
    func  = sp.lambdify(syms, expr, 'numpy')
    y_hat = np.asarray(
        func(*[X_sc[:, i] for i in range(X_sc.shape[1])]),
        dtype=float,
    ).flatten()
    return y_sc, y_hat


def compute_variable_freq(candidates):
    """Binary-presence count: how many equations contain each variable."""
    counter = Counter()
    for cand in candidates:
        _, terms, _ = extract_terms_and_ops(cand['equation'])
        for v in set(terms):
            counter[v] += 1
    return {v: counter.get(v, 0) for v in ORDERED_VARS}


def load_pareto_from_metrics(dataset_dir, val_df=None, target_col=None):
    """
    Read model_equations.csv + metrics_summary.csv from every run in
    dataset_dir, pool all rows, group by complexity, and return the
    mean NMSE_Train and NMSE_Validation at each complexity level.

    Parameters
    ----------
    dataset_dir : str
    val_df      : pd.DataFrame or None
        When provided, validation NMSE is recomputed by evaluating every
        equation on this held-out set (using each run's own scaling params),
        making the curve consistent with the bootstrap NMSE shown in panel (b).
        When None, NMSE_Validation from metrics_summary.csv is used instead
        (PySR's internal validation split — faster but a different dataset).
    target_col  : str or None  ('FValue' or 'W1') — required when val_df is set.

    Returns
    -------
    complexities  : np.ndarray  (sorted)
    train_nmse    : np.ndarray  mean NMSE_Train at each complexity
    val_nmse      : np.ndarray  mean validation NMSE at each complexity
    """

    syms    = sp.symbols("x0 x1 x2 x3 x4 x5")
    records = []

    for root, _, files in os.walk(dataset_dir):
        if 'model_equations.csv' not in files or 'metrics_summary.csv' not in files:
            continue
        try:
            eq_df  = pd.read_csv(os.path.join(root, 'model_equations.csv'))
            met_df = pd.read_csv(os.path.join(root, 'metrics_summary.csv'))
            if len(eq_df) != len(met_df):
                continue

            train_nmse = pd.to_numeric(met_df['NMSE_Train'], errors='coerce')

            if val_df is not None:
                # Recompute validation NMSE on the held-out bias_val set
                try:
                    model_eq_path = os.path.join(root, 'model_equations.csv')
                    scaling = load_scaling_params(model_eq_path)
                    X_sc, y_sc, _ = apply_scaling(val_df, target_col, scaling,
                                                   feature_cols=FEATURE_COLS)
                except Exception:
                    continue

                val_nmse_list = []
                for _, row in eq_df.iterrows():
                    try:
                        expr  = sp.sympify(row['sympy_format'])
                        func  = sp.lambdify(syms, expr, 'numpy')
                        y_hat = np.asarray(
                            func(*[X_sc[:, i] for i in range(X_sc.shape[1])]),
                            dtype=float,
                        ).flatten()
                        if not np.all(np.isfinite(y_hat)):
                            val_nmse_list.append(np.nan)
                            continue
                        var_y = np.var(y_sc)
                        nmse  = float(np.mean((y_sc - y_hat) ** 2)) / var_y \
                                if var_y > 0 else np.inf
                        val_nmse_list.append(nmse)
                    except Exception:
                        val_nmse_list.append(np.nan)
                val_nmse_col = pd.Series(val_nmse_list, dtype=float)
            else:
                val_nmse_col = pd.to_numeric(met_df['NMSE_Validation'], errors='coerce')

            combined = pd.DataFrame({
                'complexity':  pd.to_numeric(eq_df['complexity'], errors='coerce'),
                'NMSE_Train':  train_nmse,
                'NMSE_Val':    val_nmse_col,
            }).dropna()
            records.append(combined)
        except Exception:
            continue

    if not records:
        print(f"  [warning] No metrics data found in {dataset_dir}")
        return np.array([]), np.array([]), np.array([])

    all_df  = pd.concat(records, ignore_index=True)
    grouped = all_df.groupby('complexity').mean().reset_index().sort_values('complexity')

    src = 'holdout val' if val_df is not None else 'metrics_summary'
    print(f"  Loaded {len(records)} runs from {os.path.basename(dataset_dir)} [{src}]")
    return (grouped['complexity'].to_numpy(),
            grouped['NMSE_Train'].to_numpy(),
            grouped['NMSE_Val'].to_numpy())


def load_winner_pareto(winner):
    """
    Read the full Pareto frontier from a single winning run's
    model_equations.csv + metrics_summary.csv.

    Also finds the complexity of the winning equation as it appears in
    model_equations.csv (by matching the sympy_format string), so the
    vertical marker is guaranteed to land on an actual dot.

    Returns a dict with keys:
        xs, train_nmse, val_nmse  – arrays sorted by complexity
        win_complexity            – complexity of winner as stored in model_equations.csv
    """

    run_dir  = winner['run_dir']
    eq_path  = os.path.join(run_dir, 'model_equations.csv')
    met_path = os.path.join(run_dir, 'metrics_summary.csv')

    eq_df  = pd.read_csv(eq_path)
    met_df = pd.read_csv(met_path)

    combined = pd.DataFrame({
        'complexity': pd.to_numeric(eq_df['complexity'],        errors='coerce'),
        'NMSE_Train': pd.to_numeric(met_df['NMSE_Train'],       errors='coerce'),
        'NMSE_Val':   pd.to_numeric(met_df['NMSE_Validation'],  errors='coerce'),
    }).dropna().sort_values('complexity')

    xs = combined['complexity'].to_numpy()

    # winner['complexity'] is the row index written by _select_best_equation
    # (score.idxmax() returns a positional 0-based index, not the complexity
    # value).  Look up the actual complexity at that row position.
    best_row_idx = winner['complexity']
    if best_row_idx < len(eq_df):
        raw = pd.to_numeric(eq_df.iloc[best_row_idx]['complexity'], errors='coerce')
        win_complexity = int(raw) if not np.isnan(float(raw)) else int(xs[-1])
    else:
        win_complexity = int(xs[-1])

    return {
        'xs':            xs,
        'train_nmse':    combined['NMSE_Train'].to_numpy(),
        'val_nmse':      combined['NMSE_Val'].to_numpy(),
        'win_complexity': win_complexity,
    }


def _draw_winner_pareto(ax_L, ax_R, flux_data, w1_data, flux_winner, w1_winner):
    """
    Panel (a) — winner_pareto variant.
    Two sub-panels showing the full Pareto frontier from the single run that
    produced the chosen best-of-best expression (not aggregated across runs).
    Each panel has exactly two curves: Training NMSE and Validation NMSE.
    The selected expression is marked with an orange dashed vertical line.
    """
    TRAIN_COLOR = '#4C9B6F'
    VAL_COLOR   = '#84A8C5'
    BEST_COLOR  = 'darkorange'

    groups = [
        (ax_L, flux_data, flux_winner, r'$\phi_\tau$'),
        (ax_R, w1_data,   w1_winner,   r'$\Delta_\tau$'),
    ]

    for ax, data, winner, title in groups:
        if not data or len(data['xs']) == 0:
            ax.set_title(title)
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        xs = data['xs']
        train_ys = data['train_nmse']
        val_ys   = data['val_nmse']
        ax.plot(xs, train_ys, marker='o', markersize=5, linewidth=2,
                color=TRAIN_COLOR, label='Training',   alpha=0.9)
        ax.plot(xs, val_ys,   marker='s', markersize=5, linewidth=2,
                color=VAL_COLOR,   label='Validation', alpha=0.9)

        win_x = data.get('win_complexity', winner['complexity'])
        ax.axvline(win_x, color=BEST_COLOR, linestyle='--',
                   linewidth=1.8, label=f'Best (c={win_x})', zorder=4)

        # Larger highlighted dots at the winning complexity so the line
        # is visually anchored to an actual point on each curve.
        mask = xs == win_x
        if mask.any():
            ax.plot(xs[mask], train_ys[mask], marker='o', markersize=10,
                    color=BEST_COLOR, markeredgecolor='white', markeredgewidth=1,
                    zorder=5, linestyle='none')
            ax.plot(xs[mask], val_ys[mask],   marker='s', markersize=10,
                    color=BEST_COLOR, markeredgecolor='white', markeredgewidth=1,
                    zorder=5, linestyle='none')

        ax.set_yscale('log')
        ax.set_xlabel('Complexity')
        ax.set_ylabel('NMSE')
        ax.set_title(title, fontsize=20, pad=8)
        ax.grid(True, which='both', linestyle='--', alpha=0.35)
        ax.legend(fontsize=11)


def equation_to_latex(eq_str):
    """Sympy-format equation string -> matplotlib mathtext LaTeX string."""
    x0, x1, x2, x3, x4, x5 = sp.symbols("x0 x1 x2 x3 x4 x5")
    gamma_s   = sp.Symbol('gamma')
    lambda_s  = sp.Symbol('lamda')
    delta_s   = sp.Symbol('delta')
    epsilon_s = sp.Symbol('epsilon')
    N1_s      = sp.Symbol('N_1')
    N2_s      = sp.Symbol('N_2')
    try:
        expr = sp.sympify(eq_str)
        expr = expr.subs(
            [(x0, gamma_s), (x1, lambda_s), (x2, delta_s),
             (x3, epsilon_s), (x4, N1_s), (x5, N2_s)],
            simultaneous=True,
        )
        latex_str = sp.latex(expr)
        latex_str = latex_str.replace('lamda', r'\lambda')
        def _round_coeff(m):
            return f'{float(m.group(0)):.3g}'
        return re.sub(r'\d+\.\d+', _round_coeff, latex_str)
    except Exception as exc:
        print(f"  [latex fallback] {exc}")
        s = eq_str
        for token, repl in [
            ('x5','N_2'),('x4','N_1'),('x3',r'\varepsilon'),
            ('x2',r'\delta'),('x1',r'\lambda'),('x0',r'\gamma'),
        ]:
            s = s.replace(token, repl)
        return s


# ===========================================================================
# PANEL DRAWING FUNCTIONS
# ===========================================================================
def _draw_complexity_vs_nmse(ax, ranked, flux_winner, w1_winner):
    """
    Panel (a): complexity vs bootstrap NMSE for all bias runs.
    phi_tau runs = blue circles, Delta_tau runs = red squares.
    The 1.5x-rule winner for each output is highlighted with a gold border.
    """
    flux_runs = [r for r in ranked if r['target'] == 'FValue']
    w1_runs   = [r for r in ranked if r['target'] == 'W1']

    def _plot_series(runs, winner, color, marker):
        runs_sorted = sorted(runs, key=lambda r: r.get('true_complexity', r['complexity']))
        xs = [r.get('true_complexity', r['complexity']) for r in runs_sorted]
        ys = [r['mean_nmse']  for r in runs_sorted]
        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.5, zorder=2)
        for r in runs_sorted:
            is_win = r is winner
            ax.errorbar(
                r.get('true_complexity', r['complexity']), r['mean_nmse'], yerr=r['std_nmse'],
                fmt=marker,
                color=color,
                markersize=10 if is_win else 6,
                markerfacecolor=color,
                markeredgecolor=WIN_EDGE if is_win else color,
                markeredgewidth=2.5 if is_win else 0.5,
                ecolor=color, elinewidth=1, capsize=3, alpha=0.85,
                zorder=3,
            )

    _plot_series(flux_runs, flux_winner, FLUX_COLOR, 'o')
    _plot_series(w1_runs,   w1_winner,   W1_COLOR,   's')

    # Mark the winners explicitly in the legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=FLUX_COLOR,
               markeredgecolor=FLUX_COLOR, markersize=8, label=r'$\phi_\tau$ runs'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=W1_COLOR,
               markeredgecolor=W1_COLOR,   markersize=8, label=r'$\Delta_\tau$ runs'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='grey',
               markeredgecolor=WIN_EDGE, markeredgewidth=2.5,
               markersize=10, label='Best (of the best)'),
    ]
    ax.legend(handles=legend_elems, fontsize=13, loc='upper right')

    ax.set_xlabel('Complexity')
    ax.set_ylabel('Bootstrap NMSE')
    ax.set_title('Complexity vs NMSE of "Best" Eqns.', pad=8)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.set_ylim(bottom=0)


def _draw_complexity_vs_nmse_split(ax_L, ax_R, ranked, flux_winner, w1_winner):
    """
    Panel (a) — split variant: two side-by-side sub-panels, one per output,
    mirroring the layout of panel (c).  Each panel shows complexity vs
    bootstrap NMSE with a connecting line and the 1.5x-rule winner highlighted.
    """
    groups = [
        (ax_L, [r for r in ranked if r['target'] == 'FValue'],
         flux_winner, FLUX_COLOR, 'o', r'$\phi_\tau$'),
        (ax_R, [r for r in ranked if r['target'] == 'W1'],
         w1_winner,   W1_COLOR,   's', r'$\Delta_\tau$'),
    ]

    for ax, runs, winner, color, marker, title in groups:
        runs_sorted = sorted(runs, key=lambda r: r.get('true_complexity', r['complexity']))
        xs = [r.get('true_complexity', r['complexity']) for r in runs_sorted]
        ys = [r['mean_nmse']  for r in runs_sorted]

        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.5, zorder=2)

        for r in runs_sorted:
            is_win = r is winner
            ax.errorbar(
                r.get('true_complexity', r['complexity']), r['mean_nmse'], yerr=r['std_nmse'],
                fmt=marker, color=color,
                markersize=10 if is_win else 6,
                markerfacecolor=color,
                markeredgecolor=WIN_EDGE if is_win else color,
                markeredgewidth=2.5 if is_win else 0.5,
                ecolor=color, elinewidth=1, capsize=3, alpha=0.85,
                zorder=3,
            )

        ax.set_xlabel('Complexity')
        ax.set_ylabel('Bootstrap NMSE')
        ax.set_title(title, fontsize=20, pad=8)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.set_ylim(bottom=0)

        win_tc = winner.get('true_complexity', winner['complexity'])
        ax.annotate(
            f"selected\nc={win_tc}",
            xy=(win_tc, winner['mean_nmse']),
            xytext=(8, 12), textcoords='offset points',
            fontsize=11, color='dimgrey',
            arrowprops=dict(arrowstyle='->', color='dimgrey', lw=0.8),
        )


def _draw_full_pareto(ax_L, ax_R,
                      flux_xs, flux_train, flux_val,
                      w1_xs,   w1_train,   w1_val,
                      flux_winner, w1_winner):
    """
    Panel (a) — full-pareto variant.
    Two sub-panels (one per output), each with exactly two curves:
      • Training NMSE  (mean across all runs at each complexity)
      • Validation NMSE (mean across all runs at each complexity)
    Plus an orange crosshair marking the selected winner.
    """
    TRAIN_COLOR = '#4C9B6F'
    VAL_COLOR   = '#84A8C5'
    BEST_COLOR  = 'darkorange'

    groups = [
        (ax_L, flux_xs, flux_train, flux_val, flux_winner, r'$\phi_\tau$'),
        (ax_R, w1_xs,   w1_train,   w1_val,   w1_winner,   r'$\Delta_\tau$'),
    ]

    for ax, xs, train_ys, val_ys, winner, title in groups:
        if len(xs) == 0:
            ax.set_title(title)
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        ax.plot(xs, train_ys, marker='o', markersize=5, linewidth=2,
                color=TRAIN_COLOR, label='Training', alpha=0.9)
        ax.plot(xs, val_ys,   marker='s', markersize=5, linewidth=2,
                color=VAL_COLOR,   label='Validation', alpha=0.9)

        win_x     = winner['complexity']
        win_x_lbl = winner.get('true_complexity', win_x)
        ax.axvline(win_x, color=BEST_COLOR, linestyle='--',
                   linewidth=1.8, label=f'Best (c={win_x_lbl})', zorder=4)

        ax.set_yscale('log')
        ax.set_xlabel('Complexity')
        ax.set_ylabel('NMSE')
        ax.set_title(title, fontsize=20, pad=8)
        ax.grid(True, which='both', linestyle='--', alpha=0.35)
        ax.legend(fontsize=11)


def _draw_equations(ax, flux_winner, w1_winner, flux_latex, w1_latex):
    """
    Panel (b): two selected equations as LaTeX, stacked vertically.
    """
    ax.axis('off')
    ax.text(0.50, 0.72,
            r'$\phi_\tau = ' + flux_latex + r'$',
            transform=ax.transAxes,
            ha='center', va='center', fontsize=24,
            wrap=False)
    ax.text(0.50, 0.28,
            r'$\Delta_\tau = ' + w1_latex + r'$',
            transform=ax.transAxes,
            ha='center', va='center', fontsize=29)

    # Thin divider between the two equations
    ax.axhline(0.50, color='lightgrey', linewidth=0.8, xmin=0.05, xmax=0.95)

    # Sub-labels
    ax.text(0.02, 0.95, r'$\phi_\tau$',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=18, color=FLUX_COLOR, fontstyle='italic')
    ax.text(0.02, 0.48, r'$\Delta_\tau$',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=18, color=W1_COLOR, fontstyle='italic')

    # Complexity / NMSE annotation for each
    ax.text(0.98, 0.95,
            f"c={flux_winner.get('true_complexity', flux_winner['complexity'])},  "
            f"Bootstrap NMSE={flux_winner['mean_nmse']:.3f}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=16, color='dimgrey')
    ax.text(0.98, 0.48,
            f"c={w1_winner.get('true_complexity', w1_winner['complexity'])},  "
            f"Bootstrap NMSE={w1_winner['mean_nmse']:.3f}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=16, color='dimgrey')


def _draw_freq_bars(ax, freq_dict, col_title, n_candidates):
    """
    Single variable-frequency bar chart (called twice for c-left and c-right).
    """
    heights = [freq_dict[v] for v in ORDERED_VARS]
    colors  = [VAR_COLORS[v] for v in ORDERED_VARS]
    labels  = [VAR_LABELS[v] for v in ORDERED_VARS]

    bars = ax.bar(labels, heights, color=colors, edgecolor='white', linewidth=0.6)
    ax.set_ylim(0, n_candidates + 1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylabel('Appearance Frequency', fontsize=18)
    ax.set_title(col_title, fontsize=20, pad=8)

    for bar, h in zip(bars, heights):
        xc = bar.get_x() + bar.get_width() / 2
        if h < 0.15 * n_candidates:
            ax.text(xc, h + 0.15, f'{h}', ha='center', va='bottom', fontsize=13)
        else:
            ax.text(xc, h / 2, f'{h}', ha='center', va='center',
                    color='white', fontsize=13, fontweight='bold')


def _draw_residuals(ax, y_true, y_pred, col_title, winner, dot_color):
    """
    Residual scatter (y_true - y_pred vs y_true) with NMSE annotation.
    """
    residuals = y_true - y_pred
    ax.scatter(y_true, residuals, alpha=0.3, s=6, color=dot_color, linewidths=0)
    ax.axhline(0, color='firebrick', linewidth=1.4, linestyle='--', zorder=3)
    ax.set_xlabel('True value', fontsize=18)
    ax.set_ylabel('Residual',   fontsize=18)
    ax.set_title(col_title, fontsize=20, pad=8)
    nmse_txt = (f"NMSE = {winner['mean_nmse']:.3f}\n"
                f"Complexity = {winner.get('true_complexity', winner['complexity'])}")
    ax.text(0.97, 0.96, nmse_txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='grey', alpha=0.85))


# ===========================================================================
# PANEL (d) VARIANTS
# Each function has the signature:  fn(ax_L, ax_R, ctx)
# ctx keys: ranked, flux_winner, w1_winner, val_df,
#           flux_y_true, flux_y_pred, w1_y_true, w1_y_pred
# ===========================================================================

def _panel_d_residuals(ax_L, ax_R, ctx):
    """Original: residual scatter (y_true - y_pred vs y_true)."""
    _draw_residuals(ax_L, ctx['flux_y_true'], ctx['flux_y_pred'],
                    r'$\phi_\tau$',   ctx['flux_winner'], FLUX_COLOR)
    _draw_residuals(ax_R, ctx['w1_y_true'],   ctx['w1_y_pred'],
                    r'$\Delta_\tau$', ctx['w1_winner'],   W1_COLOR)


def _panel_d_dots(ax_L, ax_R, ctx):
    """
    Cleveland-style dot plot: one dot per run's best equation.
    X = bootstrap NMSE, Y = run index (sorted by NMSE).
    A vertical dashed line shows the 1.5× threshold; winner is gold-outlined.
    One sub-panel per output variable.
    """
    groups = [
        (ax_L, 'FValue', ctx['flux_winner'], FLUX_COLOR, r'$\phi_\tau$'),
        (ax_R, 'W1',     ctx['w1_winner'],   W1_COLOR,   r'$\Delta_\tau$'),
    ]
    for ax, target, winner, color, title in groups:
        runs = sorted(
            [r for r in ctx['ranked'] if r['target'] == target],
            key=lambda r: r['mean_nmse'],
        )
        nmses = [r['mean_nmse'] for r in runs]
        labels = [f"c={r.get('true_complexity', r['complexity'])}" for r in runs]
        ys = range(len(runs))

        min_nmse  = min(nmses)
        threshold = min_nmse * BEST_OF_BEST_FACTOR

        ax.axvline(threshold, color='darkorange', linestyle='--',
                   linewidth=1.4, label=f'1.5× threshold', zorder=1)

        for i, (r, y) in enumerate(zip(runs, ys)):
            is_win = r is winner
            ax.plot(r['mean_nmse'], y,
                    marker='o', markersize=10 if is_win else 7,
                    color=color,
                    markeredgecolor=WIN_EDGE if is_win else color,
                    markeredgewidth=2.5 if is_win else 0.5,
                    zorder=3)
            ax.errorbar(r['mean_nmse'], y, xerr=r['std_nmse'],
                        fmt='none', ecolor=color, elinewidth=1,
                        capsize=3, alpha=0.6, zorder=2)

        ax.set_yticks(list(ys))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Bootstrap NMSE')
        ax.set_title(title, fontsize=20, pad=8)
        ax.grid(True, axis='x', linestyle='--', alpha=0.35)
        ax.legend(fontsize=10)


def _panel_d_bars(ax_L, ax_R, ctx):
    """
    Horizontal bar chart: one bar per run's best equation, sorted by NMSE.
    Bars are labeled with their complexity. Winner gets a gold edge.
    """
    groups = [
        (ax_L, 'FValue', ctx['flux_winner'], FLUX_COLOR, r'$\phi_\tau$'),
        (ax_R, 'W1',     ctx['w1_winner'],   W1_COLOR,   r'$\Delta_\tau$'),
    ]
    for ax, target, winner, color, title in groups:
        runs = sorted(
            [r for r in ctx['ranked'] if r['target'] == target],
            key=lambda r: r['mean_nmse'],
        )
        nmses  = [r['mean_nmse'] for r in runs]
        labels = [f"c={r.get('true_complexity', r['complexity'])}" for r in runs]
        ys     = np.arange(len(runs))

        bars = ax.barh(ys, nmses, color=color, alpha=0.7, height=0.6)
        for i, (bar, r) in enumerate(zip(bars, runs)):
            if r is winner:
                bar.set_edgecolor(WIN_EDGE)
                bar.set_linewidth(2.5)
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"c={r.get('true_complexity', r['complexity'])}",
                    va='center', fontsize=9, color='dimgrey')

        # 1.5× threshold line
        min_nmse  = min(nmses)
        threshold = min_nmse * BEST_OF_BEST_FACTOR
        ax.axvline(threshold, color='darkorange', linestyle='--',
                   linewidth=1.4, label=f'1.5× threshold', zorder=4)

        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Bootstrap NMSE')
        ax.set_title(title, fontsize=20, pad=8)
        ax.grid(True, axis='x', linestyle='--', alpha=0.35)
        ax.legend(fontsize=10)


def _panel_d_table(ax_L, ax_R, ctx):
    """
    Text table: each run's best equation with complexity, NMSE, and a
    shortened equation string. Winner row is highlighted.
    """
    groups = [
        (ax_L, 'FValue', ctx['flux_winner'], FLUX_COLOR, r'$\phi_\tau$'),
        (ax_R, 'W1',     ctx['w1_winner'],   W1_COLOR,   r'$\Delta_\tau$'),
    ]
    for ax, target, winner, color, title in groups:
        ax.axis('off')
        ax.set_title(title, fontsize=20, pad=8)
        runs = sorted(
            [r for r in ctx['ranked'] if r['target'] == target],
            key=lambda r: r['complexity'],
        )
        col_labels = ['c', 'NMSE', 'Equation']
        rows = []
        for r in runs:
            eq = r['equation']
            eq_short = (eq[:28] + '…') if len(eq) > 28 else eq
            rows.append([str(r['complexity']),
                         f"{r['mean_nmse']:.3f}",
                         eq_short])

        tbl = ax.table(
            cellText=rows,
            colLabels=col_labels,
            loc='center',
            cellLoc='left',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.35)

        # Header styling
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor('#DDDDDD')
            tbl[0, j].set_text_props(fontweight='bold')

        # Highlight winner row
        winner_idx = next(
            (i for i, r in enumerate(runs) if r is winner), None)
        if winner_idx is not None:
            for j in range(len(col_labels)):
                tbl[winner_idx + 1, j].set_facecolor('#FFF3B0')
                tbl[winner_idx + 1, j].set_text_props(fontweight='bold')


def _panel_d_pareto_scatter(ax_L, ax_R, ctx):
    """
    Complexity vs bootstrap NMSE scatter for each run's best equation.
    The 1.5× eligible region is shaded; winner is gold-outlined.
    This directly visualizes the best-of-best selection rule.
    """
    groups = [
        (ax_L, 'FValue', ctx['flux_winner'], FLUX_COLOR, r'$\phi_\tau$'),
        (ax_R, 'W1',     ctx['w1_winner'],   W1_COLOR,   r'$\Delta_\tau$'),
    ]
    for ax, target, winner, color, title in groups:
        runs = sorted(
            [r for r in ctx['ranked'] if r['target'] == target],
            key=lambda r: r.get('true_complexity', r['complexity']),
        )
        xs    = [r.get('true_complexity', r['complexity']) for r in runs]
        ys    = [r['mean_nmse']   for r in runs]
        yerrs = [r['std_nmse']    for r in runs]

        min_nmse  = min(ys)
        threshold = min_nmse * BEST_OF_BEST_FACTOR

        # Shaded eligible region
        ax.axhspan(0, threshold, color='darkorange', alpha=0.08,
                   label="'Best' Eligible", zorder=1)
        ax.axhline(threshold, color='darkorange', linestyle='--',
                   linewidth=1.4, zorder=2)
        ax.axhline(min_nmse,  color='grey',       linestyle=':',
                   linewidth=1.0, zorder=2)

        # Connecting line
        ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.4, zorder=2)

        for r in runs:
            is_win = r is winner
            ax.errorbar(
                r.get('true_complexity', r['complexity']), r['mean_nmse'], yerr=r['std_nmse'],
                fmt='o', color=color,
                markersize=10 if is_win else 6,
                markeredgecolor=WIN_EDGE if is_win else color,
                markeredgewidth=2.5 if is_win else 0.5,
                ecolor=color, elinewidth=1, capsize=3, alpha=0.85,
                zorder=3,
            )

        ax.set_xlabel('Complexity')
        ax.set_ylabel('Bootstrap NMSE')
        ax.set_title(title, fontsize=20, pad=8)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=15)


# ===========================================================================
# CONTEXT + FIGURE BUILDER
# ===========================================================================

def compute_context():
    """Load all data and run bootstrap ranking once. Returns a ctx dict."""


    print(f"Loading validation data...")
    val_df = pd.read_csv(VAL_CSV)
    print(f"  {len(val_df)} rows loaded")

    print("\n=== Collecting candidates ===")
    flux_cands = collect_best_equations(FLUX_TRAIN_DIR)
    w1_cands   = collect_best_equations(W1_TRAIN_DIR)

    print("\n=== Bootstrap ranking ===")
    ranked = rank_best_equations_by_bootstrap(
        base_directory=TRAIN_DIR,
        val_csv=VAL_CSV,
        name_filter='_only_bias',
        n_bootstrap=200,
        seed=42,
    )

    # best.txt stores the DataFrame row index (from score.idxmax()), not the PySR
    # complexity value.  Resolve the true complexity from model_equations.csv now
    # so every downstream label is correct.
    for r in ranked:
        eq_df_r  = pd.read_csv(os.path.join(r['run_dir'], 'model_equations.csv'))
        row_idx  = r['complexity']
        if row_idx < len(eq_df_r):
            raw = pd.to_numeric(eq_df_r.iloc[row_idx]['complexity'], errors='coerce')
            r['true_complexity'] = int(raw) if not np.isnan(float(raw)) else row_idx
        else:
            r['true_complexity'] = row_idx

    winners     = select_best_of_best(ranked, factor=BEST_OF_BEST_FACTOR)
    flux_winner = winners['FValue']
    w1_winner   = winners['W1']

    if W1_COMPLEXITY_OVERRIDE is not None:
        candidates = [r for r in ranked
                      if r['target'] == 'W1' and r['true_complexity'] == W1_COMPLEXITY_OVERRIDE]
        if candidates:
            w1_winner = min(candidates, key=lambda r: r['mean_nmse'])
            print(f"\n  [W1 override] using c={w1_winner['true_complexity']} "
                  f"NMSE={w1_winner['mean_nmse']:.4f}")
        else:
            print(f"\n  [W1 override] no W1 candidate found at c={W1_COMPLEXITY_OVERRIDE}, "
                  f"keeping auto selection")

    print(f"\n  phi_tau:   c={flux_winner['true_complexity']}  "
          f"NMSE={flux_winner['mean_nmse']:.4f}  {flux_winner['equation']}")
    print(f"  Delta_tau: c={w1_winner['true_complexity']}  "
          f"NMSE={w1_winner['mean_nmse']:.4f}  {w1_winner['equation']}")

    print("\nComputing residuals...")
    flux_y_true, flux_y_pred = compute_predictions(flux_winner, val_df, 'FValue')
    w1_y_true,   w1_y_pred   = compute_predictions(w1_winner,   val_df, 'W1')

    flux_freq  = compute_variable_freq(flux_cands)
    w1_freq    = compute_variable_freq(w1_cands)
    flux_latex = equation_to_latex(flux_winner['equation'])
    w1_latex   = equation_to_latex(w1_winner['equation'])
    print(f"\n  phi_tau   LaTeX: {flux_latex}")
    print(f"  Delta_tau LaTeX: {w1_latex}")

    # Pareto data for panel (a) full_pareto mode
    print("\n=== Loading Pareto metrics ===")
    _val_df_a = val_df if USE_HOLDOUT_VAL else None
    flux_pareto = load_pareto_from_metrics(
        FLUX_TRAIN_DIR, val_df=_val_df_a, target_col='FValue')
    w1_pareto   = load_pareto_from_metrics(
        W1_TRAIN_DIR,   val_df=_val_df_a, target_col='W1')

    # Pareto data for panel (a) winner_pareto mode (loaded after winners are known)
    flux_winner_pareto = load_winner_pareto(flux_winner)
    w1_winner_pareto   = load_winner_pareto(w1_winner)
    print(f"  phi_tau winner frontier:   {len(flux_winner_pareto['xs'])} complexity levels")
    print(f"  Delta_tau winner frontier: {len(w1_winner_pareto['xs'])} complexity levels")

    return dict(
        val_df=val_df,
        ranked=ranked,
        flux_winner=flux_winner,   w1_winner=w1_winner,
        flux_cands=flux_cands,     w1_cands=w1_cands,
        flux_freq=flux_freq,       w1_freq=w1_freq,
        flux_latex=flux_latex,     w1_latex=w1_latex,
        flux_y_true=flux_y_true,   flux_y_pred=flux_y_pred,
        w1_y_true=w1_y_true,       w1_y_pred=w1_y_pred,
        flux_pareto=flux_pareto,         w1_pareto=w1_pareto,
        flux_winner_pareto=flux_winner_pareto, w1_winner_pareto=w1_winner_pareto,
    )


def _panel_b_score(ax_L, ax_R, ctx):
    """
    Panel (b) — score vs complexity bar chart for each winner's run.

    PySR selects the 'best' equation as the one with the highest score on its
    Pareto frontier, where score ≡ −Δln(L)/ΔC (improvement in log-loss per
    unit of added complexity).  Complexity is measured with
    complexity_of_constants = 5 (each numeric constant costs 5 units).

    One sub-panel per target; the winning bar is highlighted in orange.
    """
    BEST_COLOR = 'darkorange'
    BAR_COLOR  = '#7090B0'

    groups = [
        (ax_L, ctx['flux_winner'], ctx['flux_winner_pareto'], r'$\phi_\tau$'),
        (ax_R, ctx['w1_winner'],   ctx['w1_winner_pareto'],   r'$\Delta_\tau$'),
    ]

    for ax, winner, pareto_data, title in groups:
        # Load score from model_equations.csv for this run
        eq_path = os.path.join(winner['run_dir'], 'model_equations.csv')
        eq_df   = pd.read_csv(eq_path)
        complexities = pd.to_numeric(eq_df['complexity'], errors='coerce')
        scores       = pd.to_numeric(eq_df['score'],      errors='coerce')
        valid        = complexities.notna() & scores.notna()
        xs    = complexities[valid].to_numpy()
        ys    = scores[valid].to_numpy()

        # Identify the winning equation's complexity using pareto_data
        win_c = pareto_data.get('win_complexity', winner['complexity'])

        colors = [BEST_COLOR if c == win_c else BAR_COLOR for c in xs]
        ax.bar(xs, ys, width=np.diff(xs, append=xs[-1]+2).clip(1, 6),
               color=colors, edgecolor='white', linewidth=0.4, alpha=0.85)

        ax.set_xlabel('Complexity')
        ax.set_ylabel('Score')
        ax.set_title(title, fontsize=20, pad=8)
        ax.grid(axis='y', linestyle='--', alpha=0.35)

        # Score formula annotation
        ax.text(0.97, 0.97,
                r'Score $\equiv -\,\dfrac{\Delta\ln L}{\Delta C}$',
                transform=ax.transAxes, fontsize=11,
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.85))

        # Mark the winning bar
        win_mask = xs == win_c
        if win_mask.any():
            win_score = ys[win_mask][0]
            ax.annotate(
                f'best\n(c={win_c})',
                xy=(win_c, win_score),
                xytext=(0, 6), textcoords='offset points',
                fontsize=9, color=BEST_COLOR, ha='center', va='bottom',
                fontweight='bold',
            )

    # Shared footnote about complexity weighting
    ax_L.figure.text(
        0.52, 0.535,
        r'$\it{complexity\_of\_constants}=5$  (each constant costs 5 units)',
        fontsize=10, ha='center', color='dimgrey', style='italic',
    )


def _bootstrap_nmse_dist(y_true, y_pred, n_bootstrap=200, seed=42):
    """Return array of per-resample NMSE values (length n_bootstrap)."""
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    out = []
    for _ in range(n_bootstrap):
        idx  = rng.choice(n, size=n, replace=True)
        mse  = np.mean((y_true[idx] - y_pred[idx]) ** 2)
        var  = np.var(y_true[idx])
        out.append(mse / var if var > 0 else np.nan)
    return np.array(out)


def _panel_b_scatter(ax_L, ax_R, ctx):
    """
    Panel (b) — predicted vs. actual scatter on the validation set.
    One sub-panel per target; diagonal = perfect prediction.
    """
    groups = [
        (ax_L, ctx['flux_y_true'], ctx['flux_y_pred'],
         ctx['flux_winner'], FLUX_COLOR, r'$\phi_\tau$'),
        (ax_R, ctx['w1_y_true'],   ctx['w1_y_pred'],
         ctx['w1_winner'],   W1_COLOR,   r'$\Delta_\tau$'),
    ]
    for ax, y_true, y_pred, winner, color, title in groups:
        ax.scatter(y_true, y_pred, color=color, alpha=0.25, s=8, rasterized=True)

        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        pad = (hi - lo) * 0.04
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                'k--', linewidth=1.2, alpha=0.6, label='y = x')
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
        ax.set_title(title, fontsize=20, pad=8)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.3)

        nmse = winner['mean_nmse']
        tc   = winner.get('true_complexity', winner['complexity'])
        ax.text(0.04, 0.96,
                f"c = {tc}\nBootstrap NMSE = {nmse:.3f}",
                transform=ax.transAxes, fontsize=11,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.85))


def _panel_b_bootstrap_dist(ax_L, ax_R, ctx):
    """
    Panel (b) — full bootstrap NMSE distribution for each winning equation.
    200 resamples of the validation set; shown as violin + strip.
    """
    groups = [
        (ax_L, ctx['flux_y_true'], ctx['flux_y_pred'],
         ctx['flux_winner'], FLUX_COLOR, r'$\phi_\tau$'),
        (ax_R, ctx['w1_y_true'],   ctx['w1_y_pred'],
         ctx['w1_winner'],   W1_COLOR,   r'$\Delta_\tau$'),
    ]
    for ax, y_true, y_pred, winner, color, title in groups:
        dist = _bootstrap_nmse_dist(y_true, y_pred, n_bootstrap=200, seed=42)

        # Violin
        parts = ax.violinplot(dist, positions=[0], widths=0.6,
                              showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.45)
            pc.set_edgecolor(color)

        # Jittered strip
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=len(dist))
        ax.scatter(jitter, dist, color=color, alpha=0.35, s=6, zorder=3)

        # Mean ± std lines
        mu, sd = dist.mean(), dist.std()
        ax.hlines(mu,      -0.25, 0.25, colors='black',     linewidth=2.0, zorder=4)
        ax.hlines(mu + sd, -0.18, 0.18, colors='dimgrey',   linewidth=1.2,
                  linestyles='--', zorder=4)
        ax.hlines(mu - sd, -0.18, 0.18, colors='dimgrey',   linewidth=1.2,
                  linestyles='--', zorder=4)

        ax.set_xticks([])
        ax.set_ylabel('Bootstrap NMSE')
        ax.set_title(title, fontsize=20, pad=8)
        ax.grid(axis='y', linestyle='--', alpha=0.35)

        tc = winner.get('true_complexity', winner['complexity'])
        ax.text(0.97, 0.97,
                f"c = {tc}\n"
                f"mean = {mu:.3f}\n"
                f"std  = {sd:.3f}",
                transform=ax.transAxes, fontsize=11,
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.85))


def make_figure(ctx, panel_d_fn, output_path):
    """Build the 2×2 figure, slot in panel_d_fn for panel (d), and save."""
    flux_winner = ctx['flux_winner']
    w1_winner   = ctx['w1_winner']

    fig = plt.figure(figsize=(16, 9))
    outer = gridspec.GridSpec(2, 2, figure=fig,
                              hspace=0.28, wspace=0.15,
                              left=0.08, right=0.97,
                              top=0.96,  bottom=0.08)

    split_a = PANEL_A_MODE in ('split_best', 'full_pareto', 'winner_pareto')
    if split_a:
        inner_A = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[0, 0], wspace=0.38)
        ax_A_L = fig.add_subplot(inner_A[0])
        ax_A_R = fig.add_subplot(inner_A[1])
    else:
        ax_A = fig.add_subplot(outer[0, 0])

    # Panel (b) — split 1×2 for the swappable panel_d_fn (e.g. pareto scatter)
    inner_B = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0, 1], wspace=0.38)
    ax_B_L = fig.add_subplot(inner_B[0])
    ax_B_R = fig.add_subplot(inner_B[1])

    inner_C = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1, 0], wspace=0.38)
    ax_C_L = fig.add_subplot(inner_C[0])
    ax_C_R = fig.add_subplot(inner_C[1])

    # Panel (d) — single axes for equations
    ax_D = fig.add_subplot(outer[1, 1])

    # Panel (a)
    if PANEL_A_MODE == 'winner_pareto':
        _draw_winner_pareto(ax_A_L, ax_A_R,
                            ctx['flux_winner_pareto'], ctx['w1_winner_pareto'],
                            flux_winner, w1_winner)
    elif PANEL_A_MODE == 'full_pareto':
        flux_xs, flux_train, flux_val = ctx['flux_pareto']
        w1_xs,   w1_train,   w1_val   = ctx['w1_pareto']
        _draw_full_pareto(ax_A_L, ax_A_R,
                          flux_xs, flux_train, flux_val,
                          w1_xs,   w1_train,   w1_val,
                          flux_winner, w1_winner)
    elif PANEL_A_MODE == 'split_best':
        _draw_complexity_vs_nmse_split(
            ax_A_L, ax_A_R, ctx['ranked'], flux_winner, w1_winner)
    else:
        _draw_complexity_vs_nmse(ax_A, ctx['ranked'], flux_winner, w1_winner)

    # Panel (b) — swappable plot (top-right)
    panel_d_fn(ax_B_L, ax_B_R, ctx)

    # Panel (c) — frequency bars
    _draw_freq_bars(ax_C_L, ctx['flux_freq'], r'$\phi_\tau$',   len(ctx['flux_cands']))
    _draw_freq_bars(ax_C_R, ctx['w1_freq'],   r'$\Delta_\tau$', len(ctx['w1_cands']))

    # Panel (d) — equations (bottom-right)
    _draw_equations(ax_D, flux_winner, w1_winner, ctx['flux_latex'], ctx['w1_latex'])

    # Remove duplicate y-labels from right sub-panels
    ax_B_R.set_ylabel('')
    ax_C_R.set_ylabel('')
    if split_a:
        ax_A_R.set_ylabel('')

    # Panel labels
    label_kw = dict(fontsize=20, fontweight='bold', va='bottom', ha='left')
    a_label_ax = ax_A_L if split_a else ax_A
    a_label_ax.text(-0.05, 1.01, '(a)', transform=a_label_ax.transAxes, **label_kw)
    ax_B_L.text(-0.05, 1.01, '(b)', transform=ax_B_L.transAxes, **label_kw)
    ax_C_L.text(-0.05, 1.01, '(c)', transform=ax_C_L.transAxes, **label_kw)
    ax_D.text(  -0.05, 1.01, '(d)', transform=ax_D.transAxes,   **label_kw)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


if __name__ == '__main__':
    ctx = compute_context()

    print("\n=== Generating primary figure ===")
    make_figure(ctx, _panel_b_scatter, OUTPUT_PATH)
