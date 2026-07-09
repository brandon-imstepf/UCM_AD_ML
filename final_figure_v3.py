"""
final_figure_v3.py
==================
Recreates the 3-panel × 2-column presentation figure for the PySR symbolic
regression results on the NTM bias training data.

Left column  : phi_tau  (FValue target, flux_only_bias runs)
Right column : Delta_tau (W1 target,   w1_only_bias runs)

Panel A  Residuals scatter of the bootstrap-selected best equation on the
         bias validation set.  Every 'best' equation from every bias run is
         tested via bootstrap NMSE; the run with the lowest mean bootstrap
         NMSE wins and its residuals are plotted.

Panel B  Normalised frequency bar chart: how often each of the six NTM
         parameters (gamma, lambda, delta, epsilon, N1, N2) appears across
         all best equations found in the bias runs.

Panel C  The winning equations rendered as LaTeX math text.

Usage
-----
    python final_figure_v3.py

Output
------
    final_figure_v3.png  saved to the UCM_AD_ML directory.
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

# ---------------------------------------------------------------------------
# Make pysr_data_processor_v2 importable (it lives in the same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pysr_data_processor_v2 import (
    load_scaling_params,            # reads scaling_params.mat from a run folder
    apply_scaling,                  # standard-scales features + target
    evaluate_single_equation_scaled,  # single-equation NMSE on scaled data
    extract_terms_and_ops,          # parses a sympy string → variable names used
    rank_best_equations_by_bootstrap,
    select_best_of_best,
)

# ===========================================================================
# PATHS  –  everything lives under sim_csv_v11
# ===========================================================================
_BASE = r'C:\Users\USER\Documents\Code\UCM_AD_ML\datasets\sim_csv_v11'

TRAIN_DIR   = os.path.join(_BASE, 'train')
E3_DATA_DIR = os.path.join(_BASE, 'e3_data')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'final_figure_v3.png')

# Validation CSV
VAL_CSV = os.path.join(E3_DATA_DIR, 'bias_val_e3.csv')

# One training-run directory per output variable (bias only)
FLUX_TRAIN_DIR = os.path.join(TRAIN_DIR, 'flux_only_bias_train_e3')
W1_TRAIN_DIR   = os.path.join(TRAIN_DIR, 'w1_only_bias_train_e3')

# Tolerance multiplier for the "best of best" selection rule (mirrors PySR default)
BEST_OF_BEST_FACTOR = 1.5

# Feature columns in the data CSVs (order must match x0..x5 in equations)
FEATURE_COLS = ['gamma1', 'lambda1', 'delta', 'epsilon', 'NRow', 'NCol']

# ===========================================================================
# AESTHETICS
# ===========================================================================
# x-axis tick labels for the bar charts (Panel B)
VAR_LABELS = {
    'x0': r'$\gamma$',
    'x1': r'$\lambda$',
    'x2': r'$\delta$',
    'x3': r'$\varepsilon$',
    'x4': r'$N_1$',
    'x5': r'$N_2$',
}

# Bar colours chosen to match the original figure palette
VAR_COLORS = {
    'x0': '#E8A0A0',   # pink-red  (gamma)
    'x1': '#A0B8D8',   # blue      (lambda)
    'x2': '#A0D8A0',   # green     (delta)
    'x3': '#C8A0D8',   # purple    (epsilon)
    'x4': '#D8D890',   # yellow    (N1)
    'x5': '#C8C8C8',   # grey      (N2)
}

ORDERED_VARS = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5']

plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        14,
    'axes.titlesize':   18,
    'axes.labelsize':   13,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  12,
})


# ===========================================================================
# STEP 1 – collect 'best' equations from best.txt files
# ===========================================================================
def collect_best_equations(dataset_dir):
    """
    Walk dataset_dir (e.g. flux_only_bias_train_e3) and read the first line of
    every best.txt found in timestamped run sub-folders.

    Line format:
        Selected Best Function: <complexity>, <sympy_equation>

    Returns
    -------
    list of dict, each with keys: run, run_dir, complexity, equation
    """
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


# ===========================================================================
# STEP 2 – bootstrap NMSE comparison to find the single best equation
# ===========================================================================
def select_best_by_bootstrap(candidates, val_df, target_col, n_bootstrap=200,
                              seed=42):
    """
    For every candidate equation:
      1. Load the run's scaling_params.mat (same folder as model_equations.csv).
      2. Apply those scaling params to the validation DataFrame.
      3. Evaluate the equation on the scaled data with bootstrap resampling.
      4. Record mean bootstrap NMSE.

    The candidate with the lowest mean NMSE is returned.

    Parameters
    ----------
    candidates   : list of dicts from collect_best_equations()
    val_df       : pd.DataFrame  – unscaled validation data
    target_col   : 'FValue' or 'W1'
    n_bootstrap  : number of bootstrap resamples per candidate
    seed         : random seed for reproducibility

    Returns
    -------
    dict  – the winning candidate, augmented with 'mean_nmse' and 'std_nmse'
    """
    np.random.seed(seed)
    best_cand  = None
    best_nmse  = np.inf

    # Pre-build sympy symbols for the 6 features
    syms = sp.symbols("x0 x1 x2 x3 x4 x5")

    for cand in candidates:
        model_eq_path = os.path.join(cand['run_dir'], 'model_equations.csv')
        if not os.path.exists(model_eq_path):
            print(f"    [skip – no model_equations.csv] {cand['run']}")
            continue

        # Load per-run scaling parameters (fitted on training data)
        try:
            scaling = load_scaling_params(model_eq_path)
        except FileNotFoundError:
            print(f"    [skip – no scaling_params.mat] {cand['run']}")
            continue

        # Scale the validation features and target using the training statistics
        try:
            X_sc, y_sc, _ = apply_scaling(
                val_df, target_col, scaling, feature_cols=FEATURE_COLS
            )
        except Exception as exc:
            print(f"    [skip – scaling error] {cand['run']}: {exc}")
            continue

        # Compile the equation into a fast numpy function
        try:
            expr  = sp.sympify(cand['equation'])
            func  = sp.lambdify(syms, expr, 'numpy')
            y_hat = np.asarray(
                func(*[X_sc[:, i] for i in range(X_sc.shape[1])]),
                dtype=float
            ).flatten()
        except Exception as exc:
            print(f"    [skip – eval error] {cand['run']}: {exc}")
            continue

        # Bootstrap NMSE: resample rows, compute MSE/var each time
        n = len(y_sc)
        boot_nmse = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            mse = np.mean((y_sc[idx] - y_hat[idx]) ** 2)
            var = np.var(y_sc[idx])
            boot_nmse.append(mse / var if var > 0 else np.inf)

        mean_nmse = float(np.nanmean(boot_nmse))
        std_nmse  = float(np.nanstd(boot_nmse))

        print(f"    {cand['run']}  NMSE={mean_nmse:.4f}±{std_nmse:.4f}"
              f"  c={cand['complexity']}  eq={cand['equation']}")

        cand['mean_nmse'] = mean_nmse
        cand['std_nmse']  = std_nmse

        if mean_nmse < best_nmse:
            best_nmse = mean_nmse
            best_cand = cand

    return best_cand


# ===========================================================================
# STEP 3 – compute y_true and y_pred for the winning equation on val data
# ===========================================================================
def compute_predictions(winner, val_df, target_col):
    """
    Reload the winner's scaling, evaluate its equation on the full validation
    set (no resampling), and return (y_true_scaled, y_pred_scaled).
    """
    model_eq_path = os.path.join(winner['run_dir'], 'model_equations.csv')
    scaling       = load_scaling_params(model_eq_path)
    X_sc, y_sc, _ = apply_scaling(val_df, target_col, scaling,
                                   feature_cols=FEATURE_COLS)

    syms  = sp.symbols("x0 x1 x2 x3 x4 x5")
    expr  = sp.sympify(winner['equation'])
    func  = sp.lambdify(syms, expr, 'numpy')
    y_hat = np.asarray(
        func(*[X_sc[:, i] for i in range(X_sc.shape[1])]),
        dtype=float
    ).flatten()

    return y_sc, y_hat


# ===========================================================================
# STEP 4 – normalised variable frequency across all best equations (Panel B)
# ===========================================================================
def compute_variable_freq(candidates):
    """
    Count how many equations contain each variable at least once
    (binary presence per equation, summed across all candidates).

    Returns a dict {var: equation_count}.
    """
    counter = Counter()
    for cand in candidates:
        _, terms, _ = extract_terms_and_ops(cand['equation'])
        for v in set(terms):   # set → each variable counted once per equation
            counter[v] += 1
    return {v: counter.get(v, 0) for v in ORDERED_VARS}


# ===========================================================================
# STEP 5 – convert equation string to a LaTeX string with Greek-letter names
# ===========================================================================
def equation_to_latex(eq_str):
    """
    Convert a sympy_format equation string (using x0..x5) to a LaTeX string
    suitable for matplotlib mathtext.

    Steps:
      1. Parse with sympy (algebraic simplification is acceptable – the result
         is mathematically identical to the original).
      2. Substitute x0..x5 with Greek-letter sympy Symbols so sp.latex()
         renders them with proper backslash macros.
      3. Post-process the LaTeX string to remove trailing zeros from float
         coefficients (e.g. '0.520000000000000' → '0.52').

    Substitution map:
        x0 -> gamma  -> \\gamma
        x1 -> lamda  -> \\lambda  (avoids Python keyword; corrected below)
        x2 -> delta  -> \\delta
        x3 -> epsilon -> \\epsilon
        x4 -> N_{1}
        x5 -> N_{2}
    """
    x0, x1, x2, x3, x4, x5 = sp.symbols("x0 x1 x2 x3 x4 x5")

    # sympy recognises these names and emits the corresponding LaTeX macros
    gamma_s   = sp.Symbol('gamma')
    lambda_s  = sp.Symbol('lamda')    # 'lambda' is a Python keyword
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
        # Fix the lambda workaround
        latex_str = latex_str.replace('lamda', r'\lambda')
        # Round all decimal coefficients to 4 significant figures and strip
        # trailing zeros (e.g. '0.04681566' → '0.04682', '0.520' → '0.52').
        def _round_coeff(m):
            val = float(m.group(0))
            return f'{val:.3g}'
        latex_str = re.sub(r'\d+\.\d+', _round_coeff, latex_str)
        return latex_str
    except Exception as exc:
        print(f"    [latex fallback] {exc}")
        # Plain string substitution as last resort
        s = eq_str
        for token, repl in [
            ('x5', 'N_2'), ('x4', 'N_1'),
            ('x3', r'\varepsilon'), ('x2', r'\delta'),
            ('x1', r'\lambda'), ('x0', r'\gamma'),
        ]:
            s = s.replace(token, repl)
        return s


# ===========================================================================
# PLOTTING HELPERS
# ===========================================================================
def _plot_residuals(ax, y_true, y_pred, col_title, winner):
    """
    Scatter plot of residuals (y_true - y_pred) vs true value.
    A red dashed line at residual=0 marks perfect prediction.
    """
    residuals = y_true - y_pred

    ax.scatter(y_true, residuals, alpha=0.3, s=6, color='steelblue',
               linewidths=0)
    ax.axhline(0, color='firebrick', linewidth=1.4, linestyle='--', zorder=3)

    ax.set_xlabel('True value (scaled)', fontsize=12)
    ax.set_ylabel('Residual (scaled)', fontsize=12)
    ax.set_title(col_title, fontsize=18, pad=10)

    # NMSE annotation box in upper-right
    nmse_txt = (f"NMSE = {winner['mean_nmse']:.3f}\n"
                f"Complexity = {winner['complexity']}")
    ax.text(0.97, 0.96, nmse_txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='grey', alpha=0.85))


def _plot_freq_bars(ax, freq_dict, col_title, n_candidates):
    """
    Bar chart counting how many equations contain each variable (Panel B).
    Each bar is coloured by VAR_COLORS; value labels sit inside tall bars
    and above short ones.
    """
    heights = [freq_dict[v] for v in ORDERED_VARS]
    colors  = [VAR_COLORS[v] for v in ORDERED_VARS]
    labels  = [VAR_LABELS[v] for v in ORDERED_VARS]

    bars = ax.bar(labels, heights, color=colors, edgecolor='white',
                  linewidth=0.6)
    ax.set_ylim(0, n_candidates + 1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylabel('Number of Equations', fontsize=12)
    ax.set_title(col_title, fontsize=18, pad=10)

    for bar, h in zip(bars, heights):
        xc = bar.get_x() + bar.get_width() / 2
        if h < 0.15 * n_candidates:
            ax.text(xc, h + 0.15, f'{h}',
                    ha='center', va='bottom', fontsize=10)
        else:
            ax.text(xc, h / 2, f'{h}',
                    ha='center', va='center',
                    color='white', fontsize=10, fontweight='bold')


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    # -----------------------------------------------------------------------
    # 1. Load validation data
    # -----------------------------------------------------------------------
    print(f"Loading validation data from:\n  {VAL_CSV}")
    val_df = pd.read_csv(VAL_CSV).dropna()
    print(f"  {len(val_df)} rows loaded\n")

    # -----------------------------------------------------------------------
    # 2. Collect candidate 'best' equations from bias runs
    # -----------------------------------------------------------------------
    print("=== Collecting phi_tau (FValue) candidates ===")
    flux_cands = collect_best_equations(FLUX_TRAIN_DIR)

    print("\n=== Collecting Delta_tau (W1) candidates ===")
    w1_cands = collect_best_equations(W1_TRAIN_DIR)

    # -----------------------------------------------------------------------
    # 3. Bootstrap-rank all candidates, then apply the 1.5× PySR rule to
    #    find the simplest equation within 1.5× of the best NMSE per output.
    # -----------------------------------------------------------------------
    print("\n=== Bootstrap-ranking all candidates ===")
    all_cands = flux_cands + w1_cands
    ranked = rank_best_equations_by_bootstrap(
        base_directory=TRAIN_DIR,
        val_csv=VAL_CSV,
        name_filter='_only_bias',
        n_bootstrap=200,
        seed=42,
        # Pass pre-collected candidates directly via internal ranked list
        # by running on an already-collected list (function re-walks, so
        # we instead rely on name_filter to restrict to bias runs only).
    )

    winners = select_best_of_best(ranked, factor=BEST_OF_BEST_FACTOR)
    flux_winner = winners['FValue']
    w1_winner   = winners['W1']

    if flux_winner is None:
        raise RuntimeError("No valid phi_tau equation found.")
    if w1_winner is None:
        raise RuntimeError("No valid Delta_tau equation found.")

    print(f"\n  phi_tau:   {flux_winner['equation']}"
          f"  (NMSE={flux_winner['mean_nmse']:.4f}, complexity={flux_winner['complexity']})")
    print(f"  Delta_tau: {w1_winner['equation']}"
          f"  (NMSE={w1_winner['mean_nmse']:.4f}, complexity={w1_winner['complexity']})")

    # -----------------------------------------------------------------------
    # 4. Full-set predictions for residuals plot
    # -----------------------------------------------------------------------
    print("\nComputing residuals for Panel A...")
    flux_y_true, flux_y_pred = compute_predictions(flux_winner, val_df, 'FValue')
    w1_y_true,   w1_y_pred   = compute_predictions(w1_winner,   val_df, 'W1')

    # -----------------------------------------------------------------------
    # 5. Variable frequencies for Panel B (aggregated across ALL bias runs)
    # -----------------------------------------------------------------------
    flux_freq = compute_variable_freq(flux_cands)
    w1_freq   = compute_variable_freq(w1_cands)

    # -----------------------------------------------------------------------
    # 6. LaTeX equation strings for Panel C
    # -----------------------------------------------------------------------
    flux_latex = equation_to_latex(flux_winner['equation'])
    w1_latex   = equation_to_latex(w1_winner['equation'])
    print(f"\n  phi_tau   LaTeX: {flux_latex}")
    print(f"  Delta_tau LaTeX: {w1_latex}")

    # -----------------------------------------------------------------------
    # 7. Build the figure
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 14))

    # 3 rows; row C is shorter (equation display only)
    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[2.8, 2.2, 1.2],
        hspace=0.50,
        wspace=0.30,
    )

    ax_A_L = fig.add_subplot(gs[0, 0])   # A-left  (phi_tau   residuals)
    ax_A_R = fig.add_subplot(gs[0, 1])   # A-right (Delta_tau residuals)
    ax_B_L = fig.add_subplot(gs[1, 0])   # B-left  (phi_tau   freq bars)
    ax_B_R = fig.add_subplot(gs[1, 1])   # B-right (Delta_tau freq bars)
    ax_C_L = fig.add_subplot(gs[2, 0])   # C-left  (phi_tau   equation)
    ax_C_R = fig.add_subplot(gs[2, 1])   # C-right (Delta_tau equation)

    # ── Panel A ─────────────────────────────────────────────────────────────
    _plot_residuals(ax_A_L, flux_y_true, flux_y_pred,
                    r'$\phi_\tau$', flux_winner)
    _plot_residuals(ax_A_R, w1_y_true,   w1_y_pred,
                    r'$\Delta_\tau$', w1_winner)

    # Row label "(A)" sits just outside the left axes
    ax_A_L.text(-0.20, 1.06, '(a)', transform=ax_A_L.transAxes,
                fontsize=24, fontweight='bold')

    # ── Panel B ─────────────────────────────────────────────────────────────
    _plot_freq_bars(ax_B_L, flux_freq, r'$\phi_\tau$',  len(flux_cands))
    _plot_freq_bars(ax_B_R, w1_freq,   r'$\Delta_\tau$', len(w1_cands))

    ax_B_L.text(-0.20, 1.06, '(b)', transform=ax_B_L.transAxes,
                fontsize=24, fontweight='bold')

    # ── Panel C – equation display ───────────────────────────────────────────
    ax_C_L.axis('off')
    ax_C_R.axis('off')

    # (c) label aligned with (a) and (b): x from ax_A_L, y from ax_C_L
    from matplotlib.transforms import blended_transform_factory
    c_transform = blended_transform_factory(ax_A_L.transAxes, ax_C_L.transAxes)
    ax_C_L.text(-0.20, 1.06, '(c)', transform=c_transform,
                fontsize=24, fontweight='bold', va='top')

    # Each equation centred in its own column axes
    ax_C_L.text(0.50, 0.50,
                r'$\phi_\tau = ' + flux_latex + r'$',
                transform=ax_C_L.transAxes,
                ha='center', va='center', fontsize=16)

    ax_C_R.text(0.50, 0.50,
                r'$\Delta_\tau = ' + w1_latex + r'$',
                transform=ax_C_R.transAxes,
                ha='center', va='center', fontsize=16)

    # ── Save and show ────────────────────────────────────────────────────────
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to:\n  {OUTPUT_PATH}")
    #plt.show()


if __name__ == '__main__':
    main()
