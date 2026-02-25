import os
import re
import pandas as pd
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
import datetime

# Ground truth equations for Imstepf functions
GROUND_TRUTH = {
    "Imstepf_F1": "x0 + x1 + x2 + x3 + x4",
    "Imstepf_F2": "x0*x1*x2*x3*x4",
    "Imstepf_F3": "x0 + x1**2 + x2**3 + x3**4 + x4**5",
    "Imstepf_F4": "sin(x0) + cos(x1*x2) + sin(x3)*cos(x4)",
    "Nguyen_F1": "x**3 + x**2 + x",
    "Nguyen_F2": "x**4 + x**3 + x**2 + x",
    "Nguyen_F3": "x**5 + x**4 + x**3 + x**2 + x",
    "Nguyen_F4": "x**6 + x**5 + x**4 + x**3 + x**2 + x",
    "Nguyen_F5": "sin(x**2) * cos(x) - 1",
    "Nguyen_F6": "sin(x) + sin(x + x**2)",
    "Nguyen_F7": "log(x + 1) + log(x**2 + 1)",
}

def find_best_equation_files(root_dir):
    """Find all 'best.txt' files in the directory."""
    best_files = []
    for root, _, files in os.walk(root_dir):
        if "best.txt" in files:
            best_files.append(os.path.join(root, "best.txt"))
    return best_files

def parse_equation_from_file(file_path):
    """Extracts the best equation from a 'best.txt' file."""
    with open(file_path, 'r') as f:
        content = f.read()
    match = re.search(r"Selected Best Function:\s*\d+,\s*(.*)", content)
    if match:
        return match.group(1).strip()
    return None

def calculate_term_match_ratio(pysr_eq_str, ground_truth_eq_str):
    """Calculates the ratio of matching terms between two equations."""
    # delegate to the more detailed matcher which also normalizes variable names
    _, _, ratio = calculate_term_match_and_ratio(pysr_eq_str, ground_truth_eq_str)
    return ratio

def analyze_feasibility(dataset_dir):
    """
    Analyzes the feasibility study results by comparing PySR-found equations
    to the ground truth at different noise levels.
    """
    best_files = find_best_equation_files(dataset_dir)
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for file_path in best_files:
        path_parts = file_path.split(os.sep)
        
        # Extract noise level, function name, and sampling method from the path
        noise_str = next((part for part in path_parts if "Percent_Noise" in part), None)
        
        func_name = None
        sampling_method = None
        for part in path_parts:
            for key in GROUND_TRUTH:
                if key in part:
                    func_name = key
                    # Extract sampling method from the same part
                    if "_LHS" in part:
                        sampling_method = "LHS"
                    elif "_Uniform" in part:
                        sampling_method = "Uniform"
                    elif "_RandSample" in part:
                        sampling_method = "RandSample"
                    break
            if func_name:
                break

        if not noise_str or not func_name or not sampling_method:
            continue

        noise_level = int(re.search(r'(\d+)_Percent_Noise', noise_str).group(1))
        
        pysr_eq_str = parse_equation_from_file(file_path)
        if not pysr_eq_str:
            continue

        ground_truth_eq_str = GROUND_TRUTH[func_name]
        match_ratio = calculate_term_match_ratio(pysr_eq_str, ground_truth_eq_str)
        
        results[sampling_method][func_name][noise_level].append(match_ratio)

    return results


def find_model_equation_files(root_dir):
    """Find all 'model_equations.csv' files in the directory."""
    model_files = []
    for root, _, files in os.walk(root_dir):
        if "model_equations.csv" in files:
            model_files.append(os.path.join(root, "model_equations.csv"))
    return model_files


def pick_top_from_model_csv(file_path):
    """Read a model_equations.csv and return the top-scoring expression and its score.

    --- MODIFIED FUNCTION ---
    This now correctly identifies 'loss'/'error' columns (lower is better)
    vs 'score'/'fitness' columns (higher is better).
    """
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None

    if df.shape[1] < 3:
        return None

    # Try to find 'sympy_format' first, fallback to column 2
    if 'sympy_format' in df.columns:
        expr_col = 'sympy_format'
    else:
        expr_col = df.columns[2]

    # --- New logic to find best score column ---
    score_col = None
    lower_is_better = False
    
    lower_keywords = ['loss', 'error', 'mse', 'nmse']
    higher_keywords = ['score', 'fitness', 'r2']

    # 1. Check for higher-is-better columns
    for col in df.columns.drop(expr_col):
        name = str(col).lower()
        if any(k in name for k in higher_keywords) and pd.api.types.is_numeric_dtype(df[col]):
            score_col = col
            lower_is_better = False
            break
    
    # 2. If none, check for lower-is-better columns
    if score_col is None:
        for col in df.columns.drop(expr_col):
            name = str(col).lower()
            if any(k in name for k in lower_keywords) and pd.api.types.is_numeric_dtype(df[col]):
                score_col = col
                lower_is_better = True
                break
    
    # 3. If none, check for generic 'metric' (assume higher-is-better)
    if score_col is None:
        for col in df.columns.drop(expr_col):
            name = str(col).lower()
            if 'metric' in name and pd.api.types.is_numeric_dtype(df[col]):
                score_col = col
                lower_is_better = False # Default assumption
                break

    # 4. If still None, pick first numeric
    if score_col is None:
        for col in df.columns.drop(expr_col):
            if pd.api.types.is_numeric_dtype(df[col]):
                score_col = col
                lower_is_better = True # Default assumption for unknown: assume it's loss/error
                break
    
    # If still None, fallback to row 0
    if score_col is None:
        top_row = df.iloc[0]
        return {'expression': str(top_row[expr_col]), 'score': None, 'row': top_row, 'lower_is_better': None}

    # 5. Choose highest/lowest scoring row based on metric type
    try:
        if lower_is_better:
            top_idx = df[score_col].idxmin() # <-- Finds LOWEST
        else:
            top_idx = df[score_col].idxmax() # <-- Finds HIGHEST
        
        top_row = df.loc[top_idx]
        return {
            'expression': str(top_row[expr_col]), 
            'score': float(top_row[score_col]), 
            'row': top_row,
            'lower_is_better': lower_is_better # Pass this info out
        }
    except Exception:
        # fallback
        top_row = df.iloc[0]
        return {'expression': str(top_row[expr_col]), 'score': None, 'row': top_row, 'lower_is_better': None}


def calculate_term_match_and_ratio(pysr_eq_str, ground_truth_eq_str):
    """Return (matching_count, incorrect_count, ratio_matching_over_truth).

    ratio_matching_over_truth = matching_terms / len(ground_truth_terms) (0..1).
    incorrect_count = number of pysr terms that are not in ground truth.
    """
    # normalize variable names: map x0,x1... -> x and y0,y1... -> y to avoid symbol-name mismatches
    def _normalize_vars(s: str) -> str:
        if not isinstance(s, str):
            return s
        s2 = re.sub(r"\bx(\d+)\b", "x", s)
        s2 = re.sub(r"\by(\d+)\b", "y", s2)
        return s2

    pysr_norm = _normalize_vars(pysr_eq_str)
    gt_norm = _normalize_vars(ground_truth_eq_str)

    # aggressively normalize numeric constants in both expressions
    try:
        pysr_norm_before = pysr_norm
        pysr_norm = normalize_constants_in_expr(pysr_norm)
        gt_norm_before = gt_norm
        gt_norm = normalize_constants_in_expr(gt_norm)
        logging.getLogger(__name__).debug("normalize_constants: pysr before='%s' after='%s'", pysr_norm_before, pysr_norm)
        logging.getLogger(__name__).debug("normalize_constants: gt before='%s' after='%s'", gt_norm_before, gt_norm)
    except Exception:
        pass

    try:
        pysr_eq_temp = sp.sympify(pysr_norm)
        pysr_eq = sp.simplify(pysr_eq_temp)
        ground_truth_eq = sp.sympify(gt_norm)

        # remove multiplicative unit factors like 1.0*x or -1.0*x -> x or -x
        def _remove_unit_multipliers(expr):
            """Return expression with explicit 1.0 or -1.0 numeric multipliers removed from Mul nodes."""
            def _transform(e):
                if not getattr(e, 'is_Mul', False):
                    return e
                sign = 1
                new_args = []
                for a in e.args:
                    try:
                        if a.is_Number:
                            f = float(a)
                            if abs(f - 1.0) < 1e-9:
                                continue
                            if abs(f + 1.0) < 1e-9:
                                sign *= -1
                                continue
                    except Exception:
                        pass
                    new_args.append(a)

                if not new_args:
                    res = sp.Integer(1)
                else:
                    res = sp.Mul(*new_args)
                return -res if sign == -1 else res

            try:
                return expr.replace(lambda e: getattr(e, 'is_Mul', False), _transform)
            except Exception:
                return expr

        pysr_eq = _remove_unit_multipliers(pysr_eq)
        ground_truth_eq = _remove_unit_multipliers(ground_truth_eq)

        # expand additive structure so products that equal sums are visible as separate terms
        try:
            pysr_eq_exp = sp.expand(pysr_eq)
        except Exception:
            pysr_eq_exp = pysr_eq
        try:
            ground_truth_eq_exp = sp.expand(ground_truth_eq)
        except Exception:
            ground_truth_eq_exp = ground_truth_eq

        pysr_terms = set(pysr_eq_exp.as_ordered_terms())
        ground_truth_terms = set(ground_truth_eq_exp.as_ordered_terms())

        # Normalize terms by removing unit coefficients (1.0 or -1.0) so '1.0*x' == 'x'
        def _normalize_term_remove_unit_coeff(t):
            try:
                # ensure multiplication form then extract coefficient and multiplicative args
                coeff, rest = sp.Mul(t).as_coeff_mul()
            except Exception:
                try:
                    coeff, rest = t.as_coeff_mul()
                except Exception:
                    return t

            # if coefficient is numeric and close to 1 or -1, drop it
            try:
                f = float(coeff)
                if abs(f - 1.0) <= 1e-6:
                    if len(rest) == 1:
                        return rest[0]
                    return sp.Mul(*rest)
                if abs(f + 1.0) <= 1e-6:
                    if len(rest) == 1:
                        return -rest[0]
                    return -sp.Mul(*rest)
            except Exception:
                pass
            return t

        pysr_terms_norm = set(_normalize_term_remove_unit_coeff(t) for t in pysr_terms)
        ground_truth_terms_norm = set(_normalize_term_remove_unit_coeff(t) for t in ground_truth_terms)

        if not ground_truth_terms:
            matching_count = 0 if pysr_terms else 1
            incorrect_count = len(pysr_terms)
            return matching_count, incorrect_count, 1.0 if not pysr_terms else 0.0

        # use normalized term sets for matching
        matching_terms = pysr_terms_norm.intersection(ground_truth_terms_norm)
        incorrect_terms = pysr_terms_norm - matching_terms

        matching_count = len(matching_terms)
        incorrect_count = len(incorrect_terms)
        ratio = matching_count / len(ground_truth_terms_norm) if ground_truth_terms_norm else 0.0

        # detailed debug logging
        logger = logging.getLogger(__name__)
        logger.debug("compare: PYSR raw='%s' normalized='%s' -> simplified=%s expanded=%s", pysr_eq_str, pysr_norm, pysr_eq, pysr_eq_exp)
        logger.debug("compare: GROUND raw='%s' normalized='%s' -> simplified=%s expanded=%s", ground_truth_eq_str, gt_norm, ground_truth_eq, ground_truth_eq_exp)
        logger.debug("terms (expanded): pysr=%s", pysr_terms)
        logger.debug("terms (expanded): ground=%s", ground_truth_terms)
        logger.debug("matching_terms=%s incorrect_terms=%s matching_count=%d incorrect_count=%d ratio=%.3f",
                     matching_terms, incorrect_terms, matching_count, incorrect_count, ratio)

        return matching_count, incorrect_count, ratio
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.exception("Failed to sympify/compare expressions: '%s' vs '%s' (%s)", pysr_eq_str, ground_truth_eq_str, e)
        return 0, 0, 0.0




def normalize_constants_in_expr(expr_str):
    """Sympify expr_str and round any Float constants to nearest integers.

    Returns string form of the modified expression. If sympify fails, returns the original string.
    """
    # SymPy-based traversal: round every Float literal to 1 decimal place (user preference)
    try:
        expr = sp.sympify(expr_str)
    except Exception:
        # if sympify fails, return original string
        return expr_str

    repl = {}
    for a in sp.preorder_traversal(expr):
        try:
            if isinstance(a, sp.Float):
                # round to 1 decimal place as requested
                rounded = round(float(a), 1)
                repl[a] = sp.Float(rounded)
        except Exception:
            continue

    if repl:
        try:
            expr2 = expr.xreplace(repl)
            logging.getLogger(__name__).debug("normalize_constants_in_expr: before=%s after=%s", expr, expr2)
            return str(expr2)
        except Exception:
            return str(expr)
    return str(expr)


def evaluate_equation_on_csv(pysr_eq_str, csv_file_path, ground_truth_func_name):
    """
    Evaluates a PySR equation string against a ground truth CSV file.

    Calculates NMSE: mean((y_true - y_pred)^2) / var(y_true)
    Returns None if evaluation fails.
    """
    try:
        # 1. Load the dataset
        df = pd.read_csv(csv_file_path)
        if 'y' not in df.columns:
            logging.warning(f"CSV {csv_file_path} has no 'y' column. Skipping evaluation.")
            return None
        
        y_true = df['y']
        
        # 2. Get expected variable names from ground truth (e.g., ['x'], ['x', 'y'], ['x0', 'x1', ...])
        gt_eq = GROUND_TRUTH[ground_truth_func_name]
        gt_vars = sorted([str(s) for s in sp.sympify(gt_eq).free_symbols])
        
        # 3. Get variable names from CSV (e.g., ['x1'], ['x1', 'x2'], ['x0', 'x1', ...])
        csv_vars = sorted([col for col in df.columns if col != 'y'])
        
        # 4. Create mapping and rename columns in memory
        if len(gt_vars) != len(csv_vars):
            logging.warning(f"Var mismatch: GT has {gt_vars} but CSV has {csv_vars}. Skipping {csv_file_path}.")
            return None
        
        var_map = dict(zip(csv_vars, gt_vars))
        df_renamed = df.rename(columns=var_map)

        # 5. Prepare the PySR equation
        
        # --- NEW REMAPPING LOGIC ---
        # Get the default PySR variable names for this problem's dimension
        # e.g., if gt_vars is ['x', 'y'], default_pysr_vars is ['x0', 'x1']
        default_pysr_vars = sorted([f"x{i}" for i in range(len(gt_vars))])

        pysr_eq_str_remapped = pysr_eq_str
        
        # Only remap if the default PySR vars (x0, x1...) don't match the gt_vars (x, y)
        if default_pysr_vars != gt_vars:
            # This is the map we need: {'x0': 'x', 'x1': 'y'}
            pysr_var_map = dict(zip(default_pysr_vars, gt_vars))
            
            # We must be careful to only replace whole variable names
            # Use regex with word boundaries (e.g., \bx0\b)
            
            # --- FIX: Moved this line *outside* the loop ---
            pysr_eq_str_remapped = pysr_eq_str
            for pysr_var, gt_var in pysr_var_map.items():
                # This now accumulates changes:
                # "x0 + x1" -> "x + x1" (first loop)
                # "x + x1" -> "x + y" (second loop)
                pysr_eq_str_remapped = re.sub(r'\b' + re.escape(pysr_var) + r'\b', gt_var, pysr_eq_str_remapped)
            
            logging.debug(f"Remapped PySR eq '{pysr_eq_str}' to '{pysr_eq_str_remapped}'")
        # --- END REMAPPING LOGIC ---

        pysr_expr = sp.sympify(pysr_eq_str_remapped)
        pysr_vars = sorted([str(s) for s in pysr_expr.free_symbols])
        
        # 6. Check if PySR eq uses variables not in the ground truth
        if not set(pysr_vars).issubset(set(gt_vars)):
            # This check is now correct: e.g., set(['x', 'y']).issubset(set(['x', 'y']))
            logging.warning(f"PySR eq {pysr_eq_str_remapped} uses unknown vars. GT expects {gt_vars}. Skipping.")
            return None
            
        func = sp.lambdify(pysr_vars, pysr_expr, 'numpy')
        
        # 7. Get input data for the function
        # Ensure we only pass the variables the function actually expects, in the correct order
        # This will now correctly ask for df_renamed['x'] and df_renamed['y']
        X_input = [df_renamed[var] for var in pysr_vars]
        
        # 8. Evaluate and calculate NMSE
        y_pred = func(*X_input)
        
        if not np.all(np.isfinite(y_pred)):
            logging.warning(f"Evaluation of {pysr_eq_str_remapped} produced non-finite numbers (NaN/Inf).")
            return np.inf # Return infinity for non-finite predictions
            
        y_var = np.var(y_true)
        if y_var < 1e-10: # Avoid division by zero if y is constant
            # This is technically MSE, not NMSE, but it's the correct behavior.
            return np.mean((y_true - y_pred)**2) 
            
        nmse = np.mean((y_true - y_pred)**2) / y_var
        return nmse

    except Exception as e:
        logging.error(f"Failed to evaluate equation '{pysr_eq_str}' on file '{csv_file_path}': {e}")
        return None


def analyze_using_model_equations(dataset_dir, out_csv=None):
    """Scan model_equations.csv files, pick top expression per file, then choose best representative per (function,noise).

    --- MODIFIED FUNCTION ---
    Now returns two DataFrames:
    1. result_df: The original summary based on match_ratio.
    2. df_cand: The full list of all candidates, including their model_score AND new 'nmse_score'.
    """
    model_files = find_model_equation_files(dataset_dir)
    candidates = []

    for file_path in model_files:
        parts = file_path.split(os.sep)
        
        # extract noise_level, func_name, sampling_method
        noise_str = next((part for part in parts if "Percent_Noise" in part), None)
        noise_level = None
        if noise_str:
            m = re.search(r'(\d+)_Percent_Noise', noise_str)
            if m:
                noise_level = int(m.group(1))

        func_name = None
        sampling_method = None
        
        keys_sorted = sorted(GROUND_TRUTH.keys(), key=len, reverse=True)
        for part in parts:
            for key in keys_sorted:
                if key in part:
                    func_name = key
                    if "_LHS" in part:
                        sampling_method = "LHS"
                    elif "_Uniform" in part:
                        sampling_method = "Uniform"
                    elif "_RandSample" in part:
                        sampling_method = "RandSample"
                    break
            if func_name:
                break

        if not func_name or not sampling_method or not noise_str:
            continue

        top = pick_top_from_model_csv(file_path)
        if not top:
            continue

        expr_raw = top.get('expression')
        try:
            expr_norm_consts = normalize_constants_in_expr(expr_raw)
            expr_simpl = str(sp.simplify(sp.sympify(expr_norm_consts)))
        except Exception:
            try:
                expr_simpl = str(sp.simplify(sp.sympify(expr_raw)))
            except Exception:
                expr_simpl = expr_raw

        gt_raw = GROUND_TRUTH[func_name]

        matching_count, incorrect_count, match_ratio = calculate_term_match_and_ratio(expr_simpl, gt_raw)

        # --- NEW: Evaluate against ground truth CSV ---
        csv_file_name = f"{func_name}_{sampling_method}.csv"
        # Assumes the CSVs are in a subfolder with the same name as the noise level (e.g., .../0_Percent_Noise/0_Percent_Noise/...)
        csv_file_path = os.path.join(dataset_dir, noise_str, noise_str, csv_file_name)
        
        nmse_score = None
        if os.path.exists(csv_file_path):
            # Use the raw equation for evaluation, as it has the correct constants
            nmse_score = evaluate_equation_on_csv(expr_raw, csv_file_path, func_name)
        else:
            logging.warning(f"Could not find dataset CSV for evaluation: {csv_file_path}")
        # --- END NEW ---

        logging.getLogger(__name__).info(
            "Candidate: func=%s noise=%s file=%s score=%s nmse=%.4f match_ratio=%.3f",
            func_name, noise_level, file_path, top.get('score'), (nmse_score if nmse_score is not None else -1), match_ratio
        )

        candidates.append({
            'function': func_name,
            'noise_level': noise_level,
            'sampling_method': sampling_method,
            'file_path': file_path,
            'expr_raw': expr_raw,
            'expr_simpl': expr_simpl,
            'model_score': top.get('score'),
            'lower_is_better': top.get('lower_is_better'), # <-- Store this
            'matching_count': matching_count,
            'incorrect_count': incorrect_count,
            'match_ratio': match_ratio,
            'nmse_score': nmse_score # <-- Store new score
        })

    if not candidates:
        print("No model_equations.csv candidates found.")
        return pd.DataFrame(), pd.DataFrame() # Return two empty DFs

    df_cand = pd.DataFrame(candidates)

    # group by function and noise_level to pick representative
    chosen_rows = []
    group_cols = ['function', 'noise_level']
    for (func, noise), group in df_cand.groupby(group_cols):
        # pick highest match_ratio, tie-breaker by model_score
        
        # Check if we are minimizing or maximizing score
        lb_val = group['lower_is_better'].iloc[0] if not group['lower_is_better'].isnull().all() else True
        score_ascending = lb_val # If lower_is_better, sort ascending (lowest score is best)

        group_sorted = group.sort_values(
            by=['match_ratio', 'model_score'], 
            ascending=[False, score_ascending], 
            na_position='last'
        )
        best = group_sorted.iloc[0]

        origin_files = ';'.join(group['file_path'].astype(str).unique())

        chosen_rows.append({
            'Function': func,
            'Noise_Level': noise,
            'Chosen_Expression': best['expr_simpl'],
            'Chosen_Expression_Raw': best['expr_raw'],
            'Matching_Count': int(best['matching_count']),
            'Incorrect_Count': int(best['incorrect_count']),
            'Match_Ratio': float(best['match_ratio']),
            'Model_Score': best['model_score'],
            'nmse_score': best['nmse_score'], # <-- Add to output
            'Origin_Files': origin_files
        })

    result_df = pd.DataFrame(chosen_rows)

    if out_csv:
        result_df.to_csv(out_csv, index=False)
        print(f"Saved chosen expressions to {out_csv}")

    return result_df, df_cand # <-- MODIFIED: Return both DFs


def configure_logging(log_file=None, level=logging.DEBUG):
    """Configure module logging: console + optional file with timestamps."""
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers if this is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

def plot_results(results, save_dir):
    """
    --- MODIFIED FUNCTION ---
    Plots the success rate of PySR for each function and noise level, separated by sampling method.
    Now groups by Function on X-axis and Noise Level as Hue.
    """
    if not results:
        print("No results to plot.")
        return

    for sampling_method, data in results.items():
        plot_data = []
        for func_name, noise_levels in data.items():
            for noise, matches in noise_levels.items():
                success_rate = np.mean(matches) if matches else 0
                plot_data.append({
                    "Function": func_name,
                    "Noise Level": noise,
                    "Term Match Accuracy": success_rate
                })
        
        if not plot_data:
            continue

        df = pd.DataFrame(plot_data)

        # --- NEW: Sort functions for a cleaner plot (Nguyen_F1, F2, F3...) ---
        def get_sort_key(func_name):
            match = re.search(r'Nguyen_F(\d+)', func_name)
            if match:
                return int(match.group(1))
            match = re.search(r'Imstepf_F(\d+)', func_name)
            if match:
                return 100 + int(match.group(1)) # Put Imstepf after Nguyen
            return 999
        
        df['sort_key'] = df['Function'].apply(get_sort_key)
        df = df.sort_values(by='sort_key').drop(columns='sort_key')
        # --- END NEW SORT ---

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # --- MODIFIED: Swapped x and hue, changed palette ---
        sns.barplot(data=df, x="Function", y="Term Match Accuracy", hue="Noise Level", ax=ax, palette="bright")
        
        ax.set_title(f"PySR Term Match Accuracy vs. Noise Level ({sampling_method})", fontsize=18)
        ax.set_xlabel("Function", fontsize=14)
        ax.set_ylabel("Term Match Accuracy", fontsize=14)
        
        # --- MODIFIED: Formatting for new hue ---
        # --- FIX: Split 'ha' from 'tick_params' ---
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        plt.setp(ax.get_xticklabels(), ha='right') # Set horizontal alignment separately
        
        handles, labels = ax.get_legend_handles_labels()
        try:
            # Format labels as percentages
            new_labels = [f"{float(l):.0f}%" for l in labels]
            ax.legend(handles=handles, labels=new_labels, title="Noise Level")
        except ValueError:
            ax.legend(title="Noise Level")
        
        # Removing the bar annotations as they don't work well with grouped bars
        # for p in ax.patches:
        # ...

        plt.tight_layout()
        
        if save_dir:
            output_path = os.path.join(save_dir, f"pysr_feasibility_results_{sampling_method}.png")
            plt.savefig(output_path, dpi=300)
            print(f"Plot saved to {output_path}")
        
        plt.show()


def plot_best_model_scores(df_cand, save_dir):
    """
    Plots the best model score (e.g., lowest loss) for each model and noise level.
    """
    if df_cand is None or df_cand.empty:
        print("No candidate data to plot for best scores.")
        return

    # Create the 'model' column (e.g., "Nguyen_F1_LHS")
    df = df_cand.copy()
    # Handle missing sampling_method
    df['sampling_method'] = df['sampling_method'].fillna('Unknown')
    df['model'] = df['function'] + '_' + df['sampling_method']
    
    # Determine if we minimize or maximize the score
    # Assume it's consistent across all files
    lower_is_better = df['lower_is_better'].iloc[0] if 'lower_is_better' in df.columns and not df['lower_is_better'].isnull().all() else True
    
    if lower_is_better:
        logging.info("Plotting best scores (assuming lower is better, e.g., 'loss')")
        best_scores = df.groupby(['model', 'noise_level'])['model_score'].min().reset_index()
        y_label = "Best Score (Lowest Loss)"
    else:
        logging.info("Plotting best scores (assuming higher is better, e.g., 'R2')")
        best_scores = df.groupby(['model', 'noise_level'])['model_score'].max().reset_index()
        y_label = "Best Score (Highest R2/Fitness)"
    
    # Filter to only LHS
    plot_df = best_scores[best_scores['model'].str.contains("_LHS")].copy()
    
    # Sort functions for a cleaner plot (Nguyen_F1, F2, F3...)
    def get_sort_key(model_name):
        match = re.search(r'Nguyen_F(\d+)', model_name)
        if match:
            return int(match.group(1))
        match = re.search(r'Imstepf_F(\d+)', model_name)
        if match:
            return 100 + int(match.group(1)) # Put Imstepf after Nguyen
        return 999
    
    plot_df['sort_key'] = plot_df['model'].apply(get_sort_key)
    plot_df = plot_df.sort_values(by='sort_key').drop(columns='sort_key')
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # --- Changed palette ---
    sns.barplot(data=plot_df, x="model", y="model_score", hue="noise_level", ax=ax, palette="bright")
    
    ax.set_title(f"Best Model Score (from PySR) vs. Noise Level", fontsize=18)
    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    plt.setp(ax.get_xticklabels(), ha='right')
    
    # Format noise_level in legend
    handles, labels = ax.get_legend_handles_labels()
    try:
        new_labels = [f"{float(l):.0f}%" for l in labels]
        ax.legend(handles=handles, labels=new_labels, title="Noise Level")
    except ValueError:
        ax.legend(title="Noise Level") # Fallback
    
    # If it's a loss, a log scale is good
    if lower_is_better:
        ax.set_yscale('log')
        ax.set_ylabel(f"{y_label} (Log Scale)", fontsize=14)

    plt.tight_layout()
    
    if save_dir:
        output_path = os.path.join(save_dir, "pysr_best_model_scores_LHS.png")
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
    
    plt.show()


def plot_best_nmse_scores(df_cand, save_dir):
    """
    Plots the best NMSE score (calculated from original CSV) for each model and noise level.
    """
    if df_cand is None or df_cand.empty or 'nmse_score' not in df_cand.columns:
        print("No candidate data to plot for NMSE scores.")
        return

    # Create the 'model' column (e.g., "Nguyen_F1_LHS")
    df = df_cand.copy()
    df['sampling_method'] = df['sampling_method'].fillna('Unknown')
    df['model'] = df['function'] + '_' + df['sampling_method']
    
    # NMSE is a loss, so lower is always better
    best_scores = df.groupby(['model', 'noise_level'])['nmse_score'].min().reset_index()
    y_label = "Best NMSE (vs. Original Data)"
    
    # Filter to only LHS
    plot_df = best_scores[best_scores['model'].str.contains("_LHS")].copy()
    
    # Sort functions for a cleaner plot (Nguyen_F1, F2, F3...)
    def get_sort_key(model_name):
        match = re.search(r'Nguyen_F(\d+)', model_name)
        if match:
            return int(match.group(1))
        match = re.search(r'Imstepf_F(\d+)', model_name)
        if match:
            return 100 + int(match.group(1)) # Put Imstepf after Nguyen
        return 999
    
    plot_df['sort_key'] = plot_df['model'].apply(get_sort_key)
    plot_df = plot_df.sort_values(by='sort_key').drop(columns='sort_key')
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # --- MODIFIED: Changed palette ---
    sns.barplot(data=plot_df, x="model", y="nmse_score", hue="noise_level", ax=ax, palette="bright")

    ax.set_title(f"NMSE vs. Noise Level", fontsize=18)
    ax.set_xlabel("", fontsize=18)
    ax.set_ylabel("NMSE", fontsize=18)

    ax.tick_params(axis='x', rotation=20, labelsize=18)
    plt.setp(ax.get_xticklabels(), ha='right')
    
    # Format noise_level in legend
    handles, labels = ax.get_legend_handles_labels()
    try:
        new_labels = [f"{float(l):.0f}%" for l in labels]
        ax.legend(handles=handles, labels=new_labels, title="Noise Level")
    except ValueError:
        ax.legend(title="Noise Level") # Fallback
    
    # NMSE is best viewed on a log scale
    ax.set_yscale('log')
    ax.set_ylabel(f"{y_label} (Log Scale)", fontsize=14)

    plt.tight_layout()
    
    if save_dir:
        output_path = os.path.join(save_dir, "pysr_best_nmse_scores_LHS.png")
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # The main directory containing the feasibility study datasets
    # This should contain subdirectories for each noise level, e.g., "10_Percent_Noise"
    #main_dataset_dir = r"C:\Users\brand\Desktop\School\Quals\data\Nguyen-RBFs-n-ns-(1,3,40)_batched"
    main_dataset_dir = r"C:\Users\brand\Desktop\School\Quals\data\Imstepf-RBFs-n-ns-(1,3,40)_batched"

    # Directory to save the output plot
    output_dir = main_dataset_dir + "/feasibility_analysis_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # configure logging for verbose debug output to console and file
    #log_fname = os.path.join(output_dir, f"feasibility_debug_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    #configure_logging(log_file=log_fname, level=logging.DEBUG)
    #logging.getLogger(__name__).info("Starting feasibility analysis: %s", main_dataset_dir)

    # Run the original analysis and plot the results
    feasibility_results = analyze_feasibility(main_dataset_dir)
    plot_results(feasibility_results, output_dir)

    # Run the new model_equations-based analysis and save chosen expressions
    chosen_csv = os.path.join(output_dir, "chosen_model_expressions.csv")
    
    # --- MODIFIED CALL ---
    chosen_df, df_cand = analyze_using_model_equations(main_dataset_dir, out_csv=chosen_csv)
    
    #if not chosen_df.empty:
    #    logging.getLogger(__name__).info("Top chosen expressions (sample):\n%s", chosen_df.head().to_string(index=False))

    # --- Plot the best internal scores from the candidate list ---
    #plot_best_model_scores(df_cand, output_dir)
    
    # --- NEW CALL ---
    # Plot the best true NMSE scores from the candidate list
    plot_best_nmse_scores(df_cand, output_dir)