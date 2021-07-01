def get_heuristic_rule(val):
    multiplier = 0
    if val < 1/15:
        multiplier = 4 / 5
    elif 1/15 < val < 1 / 2:
        multiplier = 9 / 10
    elif 1/2 < val < 2:
        multiplier = 1
    elif 2 < val < 15:
        multiplier = 11 / 10
    elif val > 15:
        multiplier = 6 / 5
    return multiplier
