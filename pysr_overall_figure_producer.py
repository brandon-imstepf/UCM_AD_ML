import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# --- Import your existing data loaders ---
from pysr_analysis_utils import find_and_load_model_data, setup_matplotlib_settings

def generate_publication_figure_v2(
    phi_dir, 
    delta_dir, 
    save_path,
    best_complexity_phi=23, 
    best_complexity_delta=28,
    save_filename="Final_PySR_Graph_v2.png"
):
    """
    Generates the composite figure using 3 SEPARATE GridSpecs with manual margins.
    Fixes cutoff labels and reduces the size of the bottom row.
    """
    
    # 1. --- SETUP STYLE ---
    setup_matplotlib_settings()
    
    colors = {
        'Train': '#E69F00',      
        'Test': '#56B4E9',       
        'Validation': '#009E73', 
        'Best': '#D55E00',       
    }
    
    bar_palette = ['#F9A8A8', '#A8D1F9', '#A8F9B6', '#E4A8F9', '#F9E4A8', '#D3D3D3']

    # 2. --- LOAD DATA ---
    print("Loading Phi Data...")
    df_phi = find_and_load_model_data(phi_dir)
    print("Loading Delta Data...")
    df_delta = find_and_load_model_data(delta_dir)

    # 3. --- DEFINE LAYOUT WITH MANUAL SPACING ---
    fig = plt.figure(figsize=(20, 22)) 
    
    # --- GRIDSPEC DEFINITIONS ---
    # Key Changes:
    # 1. 'left' increased to 0.18 everywhere to fix cutoff labels.
    # 2. Row C height halved (top 0.32 -> bottom 0.20 gives height 0.12, vs previous 0.24).
    
    # ROW A: Top section
    gs_a = gridspec.GridSpec(1, 2, figure=fig, 
                             left=0.12, right=0.98, # More left margin
                             top=0.96, bottom=0.69, 
                             wspace=0.15)

    # ROW B: Middle section
    # Large gap between A(bottom=0.69) and B(top=0.59)
    gs_b = gridspec.GridSpec(1, 2, figure=fig, 
                             left=0.12, right=0.98, # More left margin
                             top=0.59, bottom=0.35, # Adjusted bottom slightly up
                             wspace=0.15)

    # ROW C: Bottom section (Equations) - HALF SIZE
    # Tight gap between B(bottom=0.35) and C(top=0.32)
    # Height is now 0.32 - 0.20 = 0.12 (much shorter)
    gs_c = gridspec.GridSpec(1, 2, figure=fig, 
                             left=0.12, right=0.98, # More left margin
                             top=0.32, bottom=0.20, 
                             wspace=0.15)

    # Helper to draw box around a specific GridSpec
    def draw_box_for_gs(gs_obj):
        ax_box = fig.add_subplot(gs_obj[:])
        ax_box.set_xticks([])
        ax_box.set_yticks([])
        ax_box.patch.set_alpha(0) 
        for spine in ax_box.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color('#888888') 
        ax_box.set_zorder(-10) 
        return ax_box

    draw_box_for_gs(gs_a)
    draw_box_for_gs(gs_b)
    ax_box_c = draw_box_for_gs(gs_c)

    # Vertical line in Row C
    pos = ax_box_c.get_position()
    center_x = pos.x0 + pos.width / 2.0
    line = Line2D([center_x, center_x], [pos.y0, pos.y1], transform=fig.transFigure,
                  color='black', linewidth=2, zorder=20)
    fig.add_artist(line)

    # Create real axes using the specific GridSpecs
    ax_nmse_phi = fig.add_subplot(gs_a[0, 0])
    ax_nmse_delta = fig.add_subplot(gs_a[0, 1], sharey=ax_nmse_phi, sharex=ax_nmse_phi)
    
    ax_bar_phi = fig.add_subplot(gs_b[0, 0])
    ax_bar_delta = fig.add_subplot(gs_b[0, 1], sharey=ax_bar_phi)
    
    ax_eq_phi = fig.add_subplot(gs_c[0, 0])
    ax_eq_delta = fig.add_subplot(gs_c[0, 1])

    # ---------------------------------------------------------
    # ROW A: NMSE vs Complexity
    # ---------------------------------------------------------
    def plot_nmse(ax, df, best_comp, title_math, is_left_plot=True):
        numeric_cols = ['complexity', 'NMSE_Train', 'NMSE_Test', 'NMSE_Validation']
        grouped = df[numeric_cols].groupby('complexity').mean().reset_index()

        ax.plot(grouped['complexity'], grouped['NMSE_Train'], marker='o', label='Training', color=colors['Train'], lw=4, markersize=10, alpha=0.9)
        ax.plot(grouped['complexity'], grouped['NMSE_Test'], marker='s', label='Testing', color=colors['Test'], lw=4, markersize=10, alpha=0.9)
        ax.plot(grouped['complexity'], grouped['NMSE_Validation'], marker='^', label='Validation', color=colors['Validation'], lw=4, markersize=10, alpha=0.9)
        ax.axvline(x=best_comp, color=colors['Best'], linestyle='--', lw=5, label='"Best"')

        ax.set_yscale('log')
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        
        ax.set_title(title_math, pad=25, fontsize=40)
        
        # Left-Justified Complexity Label
        ax.text(0.05, -0.10, 'Complexity', transform=ax.transAxes, 
                ha='left', va='top', fontsize=35)
        
        ax.tick_params(axis='both', which='major', labelsize=32) 
    
    plot_nmse(ax_nmse_phi, df_phi, best_complexity_phi, r'$\phi_{\tau}$')
    
    # Increased labelpad to handle the larger left margin
    ax_nmse_phi.set_ylabel('Average NMSE\n(Log Scale)', fontsize=35, labelpad=30)
    ax_nmse_phi.legend(loc='upper right', frameon=True, fontsize=28)

    plot_nmse(ax_nmse_delta, df_delta, best_complexity_delta, r'$\Delta_{\tau}$', is_left_plot=False)
    plt.setp(ax_nmse_delta.get_yticklabels(), visible=False)

    # ---------------------------------------------------------
    # ROW B: Normalized Frequency
    # ---------------------------------------------------------
    data_phi = [0.79, 0.00, 0.24, 0.04, 0.75, 0.82] 
    data_delta = [0.84, 0.57, 0.33, 0.17, 0.15, 0.00]
    vars_labels = [r'$\gamma$', r'$\lambda$', r'$\delta$', r'$\epsilon$', r'$N_1$', r'$N_2$']
    x_pos = np.arange(len(vars_labels))

    def plot_bars(ax, data, title_math):
        bars = ax.bar(x_pos, data, color=bar_palette, alpha=0.9, edgecolor='gray', linewidth=2)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        ax.set_title(title_math, pad=25, fontsize=40)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(vars_labels, fontsize=36)
        ax.tick_params(axis='y', labelsize=32)
        
        for bar, val in zip(bars, data):
            height = bar.get_height()
            if height > 0.15:
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                        f'{val:.2f}', ha='center', va='center', 
                        color='black', fontweight='bold', fontsize=26)
            elif height > 0.0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', 
                        color='black', fontweight='bold', fontsize=26)

    plot_bars(ax_bar_phi, data_phi, r'$\phi_{\tau}$')
    # Increased labelpad for alignment
    ax_bar_phi.set_ylabel('Normalized\nFrequency', fontsize=35, labelpad=30)
    
    plot_bars(ax_bar_delta, data_delta, r'$\Delta_{\tau}$')
    plt.setp(ax_bar_delta.get_yticklabels(), visible=False)

    # ---------------------------------------------------------
    # ROW C: Equations
    # ---------------------------------------------------------
    for ax in [ax_eq_phi, ax_eq_delta]:
        ax.axis('off')

    # Equations centered in the new, shorter boxes
    phi_tau_latex = r'$\phi_{\tau} = $'
    ax_eq_phi.text(0.02, 0.96, phi_tau_latex, fontsize=40, ha='left', va='top')
    eq_phi_latex = r"$\frac{N_2 + N_1}{e^\gamma} \cdot 0.31 - \frac{0.48 + \gamma}{\gamma - N_2 + 0.77}$"
    ax_eq_phi.text(0.5, 0.5, eq_phi_latex, ha='center', va='center', fontsize=40, transform=ax_eq_phi.transAxes)

    Delta_tau_latex = r'$\Delta_{\tau} = $'
    ax_eq_delta.text(-0.05, 0.95, Delta_tau_latex, fontsize=40, ha='left', va='top')
    eq_delta_latex = r"$(1.58 \cdot e^\gamma)^{-(5.2+\delta+\lambda)} - 0.84^{(\lambda+\gamma-\epsilon)}$"
    ax_eq_delta.text(0.46, 0.5, eq_delta_latex, ha='center', va='center', fontsize=32, transform=ax_eq_delta.transAxes)

    # ---------------------------------------------------------
    # ROW LABELS (A, B, C)
    # ---------------------------------------------------------
    # X coordinate changed to 0.02 to sit nicely in the new 0.18 margin
    # Y coordinates matched to the 'top' of each GridSpec defined above
    fig.text(0.02, 0.96, "(A)", fontsize=48, fontweight='bold', va='top')
    fig.text(0.02, 0.59, "(B)", fontsize=48, fontweight='bold', va='top')
    fig.text(0.02, 0.32, "(C)", fontsize=48, fontweight='bold', va='top')

    # ---------------------------------------------------------
    # SAVING
    # ---------------------------------------------------------
    
    full_path = os.path.join(save_path, save_filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight') 
    
    print("-" * 60)
    print(f"✅ SUCCESS! Figure saved to: {full_path}")
    print("-" * 60)
    #plt.show()

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    path_to_phi_data = r'C:\Users\brand\Desktop\School\Quals\data\train\flux_only_bias_train_e3'
    path_to_delta_data = r'C:\Users\brand\Desktop\School\Quals\data\train\w1_only_bias_train_e3'
    full_savepath = r'C:\Users\brand\Desktop\School\Quals\Presentation_Visualizations'
    
    os.makedirs(full_savepath, exist_ok=True)
    
    generate_publication_figure_v2(
        phi_dir=path_to_phi_data,
        delta_dir=path_to_delta_data,
        save_path=full_savepath,
        best_complexity_phi=23,
        best_complexity_delta=28,
    )