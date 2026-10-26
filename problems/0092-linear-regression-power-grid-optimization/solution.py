import math

PI = 3.14159

def _fluctuation(day):
    return 10 * math.sin(2 * PI * day / 10)

def power_grid_forecast(consumption_data):
    n = len(consumption_data)
    if n < 2:
        raise ValueError("Need at least 2 data points for regression")

    days = list(range(1, n + 1))

    # 1) Detrend: remove the daily fluctuation
    base = [y - _fluctuation(d) for d, y in zip(days, consumption_data)]

    # 2) Least-squares linear regression on the base data
    x_mean = sum(days) / n
    y_mean = sum(base) / n
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(days, base))
    sxx = sum((x - x_mean) ** 2 for x in days)
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean

    # 3) Predict day 15 base consumption
    target_day = 15
    base_pred = slope * target_day + intercept

    # 4) Add the day 15 fluctuation back
    forecast = base_pred + _fluctuation(target_day)

    # 5) Round, then add a 5% margin rounded up (integer math avoids float error)
    rounded = round(forecast)
    with_margin = -(-rounded * 105 // 100)  # ceil(rounded * 1.05)

    # 6) Final integer
    return int(with_margin)