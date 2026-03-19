import numpy as np
import pandas as pd

def check_distribution(arr: np.ndarray) -> bool:
    """与 rl-mining toolkit 中的分布检查保持一致。"""
    fv = arr.copy()
    fv = (fv - np.nanmean(fv)) / (np.nanstd(fv) + 1e-8)
    skew = pd.Series(fv).skew()
    kurt = pd.Series(fv).kurt()
    if np.isnan(skew) or np.isnan(kurt):
        return False
    if abs(skew) > 10 or abs(kurt) > 100:
        return False
    if np.nanstd(fv) == 0:
        return False
    return True

def check_finite_count(arr: np.ndarray, min_ratio: float = 0.95) -> bool:
    finite_ratio = np.isfinite(arr).sum() / max(1, arr.shape[0])
    return finite_ratio >= min_ratio

def check_halflife(arr: np.ndarray) -> float:
	shifted = np.roll(arr, 5)
	shifted[:5] = np.nan
	corr = finite_rcor(arr, shifted)
	return corr >= 0.5

def calc_monthlyic(df):
	values = df[['factor', 'returns']].to_numpy(dtype=float)
	f = values[:, 0]
	r = values[:, 1]
	month_codes = df.index.values.astype('datetime64[M]').astype('int64')
	unique_months, inv = np.unique(month_codes, return_inverse=True)
	sum_fr = np.bincount(inv, weights=f * r)
	sum_f2 = np.bincount(inv, weights=f * f)
	sum_r2 = np.bincount(inv, weights=r * r)
	monthly_ic_values = sum_fr / np.sqrt(sum_f2) / np.sqrt(sum_r2)

	monthly_ic = pd.Series(
		monthly_ic_values,
		index=pd.to_datetime(unique_months, unit='M').date.strftime('%Y-%m')
	)
	return monthly_ic

def calc_dailyic(df):
	values = df[['factor', 'returns']].to_numpy(dtype=float)
	f = values[:, 0]
	r = values[:, 1]
	day_codes = df.index.values.astype('datetime64[D]').astype('int64')
	unique_days, inv = np.unique(day_codes, return_inverse=True)
	sum_fr = np.bincount(inv, weights=f * r)
	sum_f2 = np.bincount(inv, weights=f * f)
	sum_r2 = np.bincount(inv, weights=r * r)
	daily_ic_values = sum_fr / np.sqrt(sum_f2) / np.sqrt(sum_r2)
	daily_ic = pd.Series(
		daily_ic_values,
		index=pd.to_datetime(unique_days, unit='D').date
	)
	return daily_ic

def r_cor(
	f1: np.ndarray, 
	f2: np.ndarray, 
	weight = None
):
	if weight is None:
		return np.dot(f1, f2) / np.sqrt(np.dot(f1, f1) * np.dot(f2, f2))
	else:
		return np.dot(weight*f1, f2) / np.sqrt(np.dot(weight*f1, weight*f1) * np.dot(weight*f2, weight*f2))

def finite_rcor(f1, f2, weight = None):
	if isinstance(f1, pd.Series):
		_f1 = f1.values
	else:
		_f1 = f1
	if isinstance(f2, pd.Series):
		_f2 = f2.values
	else:
		_f2 = f2
	mask = np.isfinite(_f1) & np.isfinite(_f2)
	return r_cor(_f1[mask], _f2[mask], weight)
	