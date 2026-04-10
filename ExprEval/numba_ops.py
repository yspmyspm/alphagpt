from __future__ import annotations

import numpy as np
from numba import njit


class TimeSeriesNumbaOps:
    @staticmethod
    @njit
    def ts_skew(x, window):
        n = len(x)
        out = np.full(n, np.nan)
        if window < 3:
            return out
        coef = (window * window) / ((window - 1) * (window - 2))
        for i in range(window - 1, n):
            y = x[i - window + 1 : i + 1]
            std = np.nanstd(y)
            mean = np.nanmean(y)
            if std <= 1e-12:
                continue
            z = (y - mean) / (std * np.sqrt(window / (window - 1)))
            out[i] = np.nanmean(z ** 3) * coef
        return out

    @staticmethod
    @njit
    def ts_argmax(x, window):
        n = len(x)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            out[i] = np.argmax(x[i - window + 1 : i + 1])
        return out + 1

    @staticmethod
    @njit
    def ts_argmin(x, window):
        n = len(x)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            out[i] = np.argmin(x[i - window + 1 : i + 1])
        return out + 1

    @staticmethod
    @njit
    def ts_mad(x, window):
        n = len(x)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            y = x[i - window + 1 : i + 1]
            out[i] = np.nanmean(np.abs(y - np.nanmean(y)))
        return out

    @staticmethod
    @njit
    def ts_kurt(x, window):
        n = len(x)
        out = np.full(n, np.nan)
        if window < 4:
            return out
        for i in range(window - 1, n):
            y = x[i - window + 1 : i + 1]
            std = np.nanstd(y)
            mean = np.nanmean(y)
            if std <= 1e-12:
                continue
            z = (y - mean) / (std * np.sqrt(window / (window - 1)))
            m4 = np.nansum(z ** 4)
            m2 = np.nansum(z ** 2)
            out[i] = (m4 * (window + 1) * window - m2 * m2 * (window - 1) * 3) / (
                (window - 1) * (window - 2) * (window - 3)
            )
        return out

    @staticmethod
    @njit
    def ts_softmax(x, window):
        n = len(x)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            y = x[i - window + 1 : i + 1].copy()
            y -= np.nanmax(y)
            y = np.exp(y)
            denom = np.nansum(y)
            if denom <= 1e-12:
                continue
            out[i] = y[-1] / denom
        return out

    @staticmethod
    @njit
    def ts_rcor(x, y, window):
        n = len(x)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            xs = x[i - window + 1 : i + 1]
            ys = y[i - window + 1 : i + 1]
            x_mod = np.sqrt(np.nansum(xs ** 2))
            y_mod = np.sqrt(np.nansum(ys ** 2))
            x_mod = max(x_mod, 1e-8)
            y_mod = max(y_mod, 1e-8)
            out[i] = np.nansum(xs * ys) / x_mod / y_mod
        return out

    @staticmethod
    @njit
    def ts_rank(x, window):
        n = len(x)
        out = np.full(n, np.nan)
        denom = max(window - 1, 1)
        for i in range(window - 1, n):
            y = x[i - window + 1 : i + 1]
            val = y[-1]
            count = np.nansum(y < val) + np.nansum(y == val) * 0.5
            out[i] = count / denom
        return out

    @staticmethod
    @njit
    def ts_wmean(x, window):
        n = len(x)
        out = np.full(n, np.nan)
        weights = np.arange(1, window + 1, dtype=np.float64)
        weight_sum = np.nansum(weights)
        for i in range(window - 1, n):
            y = x[i - window + 1 : i + 1]
            out[i] = np.nansum(y * weights) / weight_sum
        return out

    @staticmethod
    @njit
    def ts_reg_beta(x, window):
        n = len(x)
        out = np.full(n, np.nan)
        t = np.arange(window, dtype=np.float64)
        t_mean = np.nanmean(t)
        t_var = np.nanmean((t - t_mean) ** 2)
        for i in range(window - 1, n):
            y = x[i - window + 1 : i + 1]
            y_mean = np.nanmean(y)
            numerator = np.nanmean((y - y_mean) * (t - t_mean))
            out[i] = numerator / max(t_var, 1e-12)
        return out

    @staticmethod
    @njit
    def ts_reg_resid(x, window):
        n = len(x)
        out = np.full(n, np.nan)
        t = np.arange(window, dtype=np.float64)
        t_mean = np.nanmean(t)
        t_var = np.nanmean((t - t_mean) ** 2)
        t_last = float(window - 1)
        for i in range(window - 1, n):
            y = x[i - window + 1 : i + 1]
            y_mean = np.nanmean(y)
            numerator = np.nanmean((y - y_mean) * (t - t_mean))
            beta = numerator / max(t_var, 1e-12)
            alpha = y_mean - beta * t_mean
            out[i] = y[-1] - (alpha + beta * t_last)
        return out
