def estimate_risk(distance, velocity):
    if velocity <= 0:
        return "low"
    ttc = distance / velocity if velocity > 0 else float('inf')
    if ttc < 2:
        return "high"
    elif ttc < 4:
        return "medium"
    else:
        return "low"