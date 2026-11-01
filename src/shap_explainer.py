"""
shap_explainer.py
Generates local SHAP explanations for a single prediction using the XGBoost model.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Human-readable display names for sanitized feature names
FEATURE_DISPLAY_NAMES = {
    'Air temperature K'    : 'Air Temp [K]',
    'Process temperature K': 'Process Temp [K]',
    'Rotational speed rpm' : 'Rotational Speed [rpm]',
    'Torque Nm'            : 'Torque [Nm]',
    'Tool wear min'        : 'Tool Wear [min]',
    'Temp_Delta'           : 'Temp Delta (Proc−Air)',
    'Power_W'              : 'Mechanical Power [W]',
    'Wear_per_Torque'      : 'Wear / Torque',
    'Speed_Torque_Ratio'   : 'Speed / Torque',
    'Type_H'               : 'Type = H',
    'Type_L'               : 'Type = L',
    'Type_M'               : 'Type = M',
}

FEATURE_EXPLANATIONS = {
    'Tool wear min'        : 'Higher tool wear strongly increases failure risk. Schedule tool replacement.',
    'Torque Nm'            : 'Both very high and very low torque increase failure risk.',
    'Power_W'              : 'Mechanical power = Torque × Speed. Extremes indicate PWF risk.',
    'Rotational speed rpm' : 'Low RPM combined with high torque is a common failure precursor.',
    'Temp_Delta'           : 'Low temp difference may indicate cooling system issues (HDF risk).',
    'Wear_per_Torque'      : 'High wear relative to torque indicates OSF risk.',
    'Speed_Torque_Ratio'   : 'Imbalanced speed-torque ratio reflects drivetrain stress.',
    'Process temperature K': 'Elevated process temperature increases failure probability.',
    'Air temperature K'    : 'Air temperature affects cooling efficiency.',
    'Type_H'               : 'High-quality product type.',
    'Type_L'               : 'Low-quality product type — wider tolerance limits.',
    'Type_M'               : 'Medium-quality product type.',
}


def build_explainer(model):
    """Build and return a cached TreeExplainer for the XGBoost model."""
    if not SHAP_AVAILABLE:
        return None
    return shap.TreeExplainer(model)


def explain(explainer, feature_df: pd.DataFrame) -> dict:
    """
    Compute local SHAP values for one prediction.

    Returns
    -------
    dict with keys:
        available     : bool
        shap_values   : np.ndarray  (1 x n_features)
        base_value    : float
        feature_names : list[str]   (sanitized)
        display_names : list[str]   (human-readable)
        top5          : list[dict]  (sorted by |SHAP|)
    """
    result = {"available": False}
    if not SHAP_AVAILABLE or explainer is None:
        return result
    try:
        sv = explainer.shap_values(feature_df)
        # For binary classifiers, shap_values may return list [class0, class1]
        if isinstance(sv, list):
            sv = sv[1]
        sv_flat = sv[0]

        bv = explainer.expected_value
        if isinstance(bv, (list, np.ndarray)):
            bv = float(bv[1])
        else:
            bv = float(bv)

        feat_names = list(feature_df.columns)
        disp_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in feat_names]

        # Top 5 by absolute SHAP value
        order = np.argsort(np.abs(sv_flat))[::-1]
        top5 = []
        for idx in order[:5]:
            fname = feat_names[idx]
            top5.append({
                "feature"      : fname,
                "display"      : disp_names[idx],
                "shap_value"   : float(sv_flat[idx]),
                "feature_value": float(feature_df.iloc[0, idx]),
                "direction"    : "↑ Increases" if sv_flat[idx] > 0 else "↓ Decreases",
                "explanation"  : FEATURE_EXPLANATIONS.get(fname, ""),
            })

        result = {
            "available"    : True,
            "shap_values"  : sv_flat,
            "base_value"   : bv,
            "feature_names": feat_names,
            "display_names": disp_names,
            "top5"         : top5,
            "feature_df"   : feature_df,
        }
    except Exception as e:
        result["error"] = str(e)
    return result


"""def plot_waterfall(shap_result: dict) -> bytes:
    Return a waterfall plot PNG as bytes.
    if not shap_result.get("available"):
        return None
    try:
        exp = shap.Explanation(
            values      = shap_result["shap_values"],
            base_values = shap_result["base_value"],
            data        = shap_result["feature_df"].iloc[0].values,
            feature_names = shap_result["display_names"],
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(exp, show=False, max_display=12)
        plt.title("SHAP Waterfall — Local Prediction Explanation", fontsize=12, pad=10)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                    facecolor='#1a1a2e')
        plt.close()
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None"""

def plot_waterfall(shap_result: dict) -> bytes:
    """Return a waterfall plot PNG with white feature labels."""
    if not shap_result.get("available"):
        return None

    try:
        exp = shap.Explanation(
            values=shap_result["shap_values"],
            base_values=shap_result["base_value"],
            data=shap_result["feature_df"].iloc[0].values,
            feature_names=shap_result["display_names"],
        )

        shap.plots.waterfall(exp, show=False, max_display=12)

        # Get the actual figure generated by SHAP
        fig = plt.gcf()

        # Render before modifying text
        fig.canvas.draw()

        # Change ALL text elements to white
        for axis in fig.axes:

            # Feature labels and axis tick labels
            axis.tick_params(axis='both', colors='white')

            for label in axis.get_yticklabels():
                label.set_color('white')

            for label in axis.get_xticklabels():
                label.set_color('white')

            # SHAP feature names and annotations
            for txt in axis.texts:
                txt.set_color('white')

            # Axis labels
            axis.xaxis.label.set_color('white')
            axis.yaxis.label.set_color('white')

            # White axis borders
            for spine in axis.spines.values():
                spine.set_color('#666666')

        # Ensure all figure text is white
        for txt in fig.findobj(match=matplotlib.text.Text):
            txt.set_color('white')

        # Title
        fig.axes[0].set_title(
            "SHAP Waterfall — Local Prediction Explanation",
            fontsize=12,
            color='white',
            pad=10
        )

        # Background
        fig.patch.set_facecolor('#1a1a2e')

        for axis in fig.axes:
            axis.set_facecolor('#ffffff')

        plt.tight_layout()

        buf = io.BytesIO()

        plt.savefig(
            buf,
            format='png',
            dpi=110,
            bbox_inches='tight',
            facecolor='#1a1a2e'
        )

        plt.close(fig)
        buf.seek(0)

        return buf.getvalue()

    except Exception as e:
        print(f"Waterfall plot error: {e}")
        return None

def plot_bar(shap_result: dict) -> bytes:
    """Return a horizontal bar chart of SHAP contributions as bytes."""
    if not shap_result.get("available"):
        return None
    try:
        sv    = shap_result["shap_values"]
        names = shap_result["display_names"]
        order = np.argsort(sv)
        sv_s  = sv[order]
        nm_s  = [names[i] for i in order]
        colors = ['#F44336' if v > 0 else '#4CAF50' for v in sv_s]

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        ax.barh(nm_s, sv_s, color=colors, edgecolor='none', height=0.6)
        ax.axvline(0, color='white', lw=0.8, alpha=0.5)
        ax.set_xlabel('SHAP Value (contribution to failure probability)',
                       color='white', fontsize=9)
        ax.set_title('SHAP Contribution — Red = increases failure risk',
                      color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                    facecolor='#1a1a2e')
        plt.close()
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None
