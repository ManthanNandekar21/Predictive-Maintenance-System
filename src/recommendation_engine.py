"""
recommendation_engine.py
Generates context-aware, sensor-driven maintenance recommendations.
"""
import numpy as np

PRIORITY_COLORS = {
    "Low"     : "#4CAF50",
    "Medium"  : "#FFC107",
    "High"    : "#FF9800",
    "Critical": "#F44336",
}

# Physical thresholds derived from AI4I dataset statistics
THRESHOLDS = {
    "tool_wear_warn"    : 180,   # min  — warning
    "tool_wear_crit"    : 240,   # min  — critical
    "temp_delta_low"    : 8.6,   # K    — HDF risk
    "power_low"         : 3500,  # W    — PWF risk (low side)
    "power_high"        : 9000,  # W    — PWF risk (high side)
    "wear_torque_high"  : 4.0,   # ratio— OSF risk
}


def _tw_recs(tool_wear, torque):
    recs = ["Inspect cutting tools for wear and surface degradation."]
    if tool_wear > THRESHOLDS["tool_wear_crit"]:
        recs.append(f"Tool wear ({tool_wear:.0f} min) is above critical limit — schedule immediate replacement.")
    elif tool_wear > THRESHOLDS["tool_wear_warn"]:
        recs.append(f"Tool wear ({tool_wear:.0f} min) is elevated — plan replacement within the next maintenance window.")
    recs.append("Log tool wear history to refine replacement intervals.")
    return recs


def _hdf_recs(temp_delta, rpm):
    recs = ["Inspect cooling and heat dissipation systems."]
    if temp_delta < THRESHOLDS["temp_delta_low"]:
        recs.append(f"Temperature difference is low ({temp_delta:.1f} K) — check coolant flow and ventilation.")
    recs.append("Verify that air and process temperature sensors are calibrated.")
    if rpm > 1800:
        recs.append(f"High rotational speed ({rpm:.0f} rpm) with low Temp_Delta amplifies HDF risk — consider speed reduction.")
    return recs


def _pwf_recs(power_w, rpm, torque):
    recs = ["Inspect motor, power supply, and drivetrain."]
    if power_w < THRESHOLDS["power_low"]:
        recs.append(f"Mechanical power ({power_w:.0f} W) is below normal operating range — check for torque loss or RPM drop.")
    elif power_w > THRESHOLDS["power_high"]:
        recs.append(f"Mechanical power ({power_w:.0f} W) exceeds safe operating range — reduce load or increase speed.")
    recs.append("Check electrical supply stability and motor load.")
    return recs


def _osf_recs(tool_wear, torque, wear_per_torque):
    recs = ["Inspect machine for excessive mechanical load."]
    if wear_per_torque > THRESHOLDS["wear_torque_high"]:
        recs.append(f"Wear/Torque ratio ({wear_per_torque:.2f}) is high — review operating conditions.")
    recs.append("Verify torque limits are not being exceeded for the current product type.")
    recs.append("Check for mechanical binding or excessive friction.")
    return recs


def _rnf_recs():
    return [
        "Perform a full diagnostic inspection (random failure is not predictable from sensors alone).",
        "Review machine logs and operating history for anomalies.",
        "Verify all sensor readings and connections.",
        "Conduct a general mechanical health check.",
    ]


def _general_no_failure_recs(prob_failure):
    recs = ["Continue routine monitoring."]
    if prob_failure > 0.20:
        recs.append("Failure probability is above 20% — consider increasing monitoring frequency.")
    recs.append("Record current readings as baseline for trend analysis.")
    return recs


def generate(prediction_result: dict,
             diagnosis_result: dict,
             shap_result: dict,
             inputs: dict) -> dict:
    """
    Generate maintenance recommendations.

    Parameters
    ----------
    prediction_result : output from prediction.predict()
    diagnosis_result  : output from failure_diagnosis.diagnose()
    shap_result       : output from shap_explainer.explain()
    inputs            : raw user inputs dict

    Returns
    -------
    dict with keys: priority, priority_color, actions, reasons, preventive
    """
    pred      = prediction_result["prediction"]
    prob_fail = prediction_result["prob_failure"]
    risk      = prediction_result["risk_level"]
    subtypes  = diagnosis_result.get("subtypes", [])

    air_temp   = inputs.get("air_temp", 300)
    proc_temp  = inputs.get("proc_temp", 310)
    rpm        = inputs.get("rpm", 1500)
    torque     = inputs.get("torque", 40)
    tool_wear  = inputs.get("tool_wear", 100)

    # Engineered (re-computed for display; already computed in feature_engineering)
    temp_delta         = proc_temp - air_temp
    power_w            = torque * rpm * 2 * np.pi / 60
    wear_per_torque    = tool_wear / (torque + 1e-5)

    actions    = []
    reasons    = []
    preventive = []

    if pred == 0:
        priority = "Low" if risk == "Low" else "Medium"
        actions  = _general_no_failure_recs(prob_fail)
        reasons  = [f"Model predicts No Failure (probability: {prob_fail*100:.1f}%)."]
        preventive = [
            "Follow scheduled preventive maintenance intervals.",
            "Monitor tool wear and replace before reaching critical limits.",
            "Log all sensor readings for trend detection.",
        ]
    else:
        # Failure predicted
        if risk == "Critical":
            priority = "Critical"
            actions.append("⚠️  Stop machine operation and conduct safety inspection immediately.")
        elif risk == "High":
            priority = "High"
            actions.append("Arrange maintenance inspection as soon as possible.")
        else:
            priority = "High"
            actions.append("Schedule an unplanned maintenance inspection.")

        reasons.append(f"Model predicts Failure with {prob_fail*100:.1f}% probability (Risk: {risk}).")

        if not subtypes and diagnosis_result.get("available"):
            if diagnosis_result.get("unclassified"):
                reasons.append("No specific failure sub-type was identified above the confidence threshold — performing general inspection.")
                actions.extend(_rnf_recs())
        else:
            for st in subtypes:
                if st == "TWF":
                    actions.extend(_tw_recs(tool_wear, torque))
                    reasons.append(f"Tool Wear Failure (TWF) indicated — tool wear = {tool_wear:.0f} min.")
                elif st == "HDF":
                    actions.extend(_hdf_recs(temp_delta, rpm))
                    reasons.append(f"Heat Dissipation Failure (HDF) indicated — Temp_Delta = {temp_delta:.1f} K.")
                elif st == "PWF":
                    actions.extend(_pwf_recs(power_w, rpm, torque))
                    reasons.append(f"Power Failure (PWF) indicated — Mechanical Power = {power_w:.0f} W.")
                elif st == "OSF":
                    actions.extend(_osf_recs(tool_wear, torque, wear_per_torque))
                    reasons.append(f"Overstrain Failure (OSF) indicated — Wear/Torque = {wear_per_torque:.2f}.")
                elif st == "RNF":
                    actions.extend(_rnf_recs())
                    reasons.append("Random Failure (RNF) indicated.")

        # Add SHAP-driven context
        if shap_result.get("available") and shap_result.get("top5"):
            top_feat = shap_result["top5"][0]
            if top_feat["shap_value"] > 0:
                reasons.append(
                    f"SHAP: '{top_feat['display']}' = {top_feat['feature_value']:.2f} "
                    f"is the largest contributor, increasing failure risk by {top_feat['shap_value']:.3f}."
                )

        preventive = [
            "Establish tool wear monitoring with automated alerts at 80% of the maximum allowed wear.",
            "Implement coolant temperature and flow rate monitoring.",
            "Schedule periodic calibration of torque and speed sensors.",
            "Perform root-cause analysis after every failure event and update maintenance plans.",
        ]

    # Deduplicate actions while preserving order
    seen = set()
    actions_dedup = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            actions_dedup.append(a)

    return {
        "priority"       : priority,
        "priority_color" : PRIORITY_COLORS[priority],
        "actions"        : actions_dedup,
        "reasons"        : reasons,
        "preventive"     : preventive,
    }
