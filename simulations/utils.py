def model_string(type: str, ms: int, ap: float=0, fnr: float=0, fpr: float=0, pheavy: bool=False):
    if type == "oracle":
        return f"oracle_{ms}KB"
    elif type == "random":
        return "random"
    elif type == "baseline":
        return "baseline"
    elif type == "synth" and ap:
        return f"sim_ap{ap}_{ms}KB"
    elif type == "synth":
        return f"sim_fnr{fnr}_fpr{fpr}_{ms}KB" if not pheavy else "pheavy"
    else:
        return type
