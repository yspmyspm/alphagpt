import numpy as np
import pandas as pd

from configs import ModelConfig

def check_distribution(arr: np.ndarray) -> bool:
	"""与 rl-mining toolkit 中的分布检查保持一致。"""
	fv = arr.copy()
	fv = (fv - np.nanmean(fv)) / (np.nanstd(fv) + 1e-8)
	skew = pd.Series(fv).skew()
	kurt = pd.Series(fv).kurt()
	if np.isnan(skew) or np.isnan(kurt):
		return False
	if (
		abs(skew) > float(ModelConfig.DISTRIBUTION_CHECK_SKEW_LIMIT)
		or abs(kurt) > float(ModelConfig.DISTRIBUTION_CHECK_KURT_LIMIT)
	):
		return False
	if np.nanstd(fv) == 0:
		return False
	return True


def score_finite_ratio(arr: np.ndarray, min_ratio: float | None = None) -> float:
	"""返回 [0, 1] 的有限值比例分数，1 表示完全通过。"""
	if min_ratio is None:
		min_ratio = float(ModelConfig.SCORE_FINITE_MIN_RATIO)
	finite_ratio = np.isfinite(arr).sum() / max(1, arr.shape[0])
	return min(1.0, finite_ratio / min_ratio)


def score_distribution(
	arr: np.ndarray,
	skew_limit: float | None = None,
	kurt_limit: float | None = None,
) -> float:
	"""返回 [0, 1] 的分布分数，1 表示完全通过。"""
	if skew_limit is None:
		skew_limit = float(ModelConfig.SCORE_DISTRIBUTION_SKEW_LIMIT)
	if kurt_limit is None:
		kurt_limit = float(ModelConfig.SCORE_DISTRIBUTION_KURT_LIMIT)
	fv = arr.copy()
	fv = (fv - np.nanmean(fv)) / (np.nanstd(fv) + 1e-8)
	skew = pd.Series(fv).skew()
	kurt = pd.Series(fv).kurt()
	if np.isnan(skew) or np.isnan(kurt):
		return 0.0
	if np.nanstd(fv) == 0:
		return 0.0
	# 线性插值：skew 在 [0, 10] 得 1，在 [10, 20] 线性降到 0
	skew_score = max(0.0, 1.0 - abs(skew) / skew_limit)
	kurt_score = max(0.0, 1.0 - abs(kurt) / kurt_limit)
	return min(skew_score, kurt_score)


def score_halflife(arr: np.ndarray, min_corr: float | None = None, lag: int | None = None) -> float:
	"""返回 [0, 1] 的半衰期分数，corr >= min_corr 得 1。"""
	if min_corr is None:
		min_corr = float(ModelConfig.SCORE_HALFLIFE_MIN_CORR)
	if lag is None:
		lag = int(ModelConfig.HALFLIFE_LAG)
	shifted = np.roll(arr, lag)
	shifted[:lag] = np.nan
	corr = finite_rcor(arr, shifted)
	if np.isnan(corr) or np.isinf(corr):
		return 0.0
	
	return min(1.0, max(0.0, (corr + 1.0) / (min_corr + 1.0)))  # 从 -1 到 min_corr 线性映射到 [0,1]


def check_finite_count(arr: np.ndarray, min_ratio: float | None = None) -> bool:
	if min_ratio is None:
		min_ratio = float(ModelConfig.SCORE_FINITE_MIN_RATIO)
	finite_ratio = np.isfinite(arr).sum() / max(1, arr.shape[0])
	return finite_ratio >= min_ratio

def check_halflife(arr: np.ndarray) -> float:
	lag = int(ModelConfig.HALFLIFE_LAG)
	shifted = np.roll(arr, lag)
	shifted[:lag] = np.nan
	corr = finite_rcor(arr, shifted)
	return corr >= float(ModelConfig.SCORE_HALFLIFE_MIN_CORR)

def calc_monthlyic(df):
	df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
	values = df[['factor', 'returns']].to_numpy(dtype=float)
	f = values[:, 0]
	r = values[:, 1]
	month_codes = df.index.values.astype('datetime64[M]').astype('int64')
	unique_months, inv = np.unique(month_codes, return_inverse=True)
	sum_fr = np.bincount(inv, weights=f * r)
	sum_f2 = np.bincount(inv, weights=f * f)
	sum_r2 = np.bincount(inv, weights=r * r)

	denom = np.sqrt(sum_f2) * np.sqrt(sum_r2)
	monthly_ic_values = np.divide(
		sum_fr,
		denom,
		out=np.full_like(sum_fr, np.nan, dtype=float),
		where=(denom != 0) & np.isfinite(denom)
	)

	monthly_ic = pd.Series(
		monthly_ic_values,
		index=pd.to_datetime(unique_months, unit='M').strftime('%Y-%m')
	)
	return monthly_ic

def calc_dailyic(df):
	df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
	values = df[['factor', 'returns']].to_numpy(dtype=float)
	f = values[:, 0]
	r = values[:, 1]
	day_codes = df.index.values.astype('datetime64[D]').astype('int64')
	unique_days, inv = np.unique(day_codes, return_inverse=True)
	sum_fr = np.bincount(inv, weights=f * r)
	sum_f2 = np.bincount(inv, weights=f * f)
	sum_r2 = np.bincount(inv, weights=r * r)
	
	denom = np.sqrt(sum_f2) * np.sqrt(sum_r2)

	daily_ic_values = np.divide(
		sum_fr,
		denom,
		out=np.full_like(sum_fr, np.nan, dtype=float),
		where=(denom != 0) & np.isfinite(denom)
	)

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
	if isinstance(f1, pd.Series):
		_f1 = f1.replace([np.inf, -np.inf, np.nan], 0).to_numpy()
	else:
		_f1 = np.copy(f1)
		_f1[~np.isfinite(_f1)] = 0
	if isinstance(f2, pd.Series):
		_f2 = f2.replace([np.inf, -np.inf, np.nan], 0).to_numpy()
	else:
		_f2 = np.copy(f2)
		_f2[~np.isfinite(_f2)] = 0


	if np.dot(_f1, _f1) == 0:
		return 0.0
	if np.dot(_f2, _f2) == 0:
		return 0.0
	
	
	
	if weight is None:
		return np.dot(_f1, _f2) / np.sqrt(np.dot(_f1, _f1) * np.dot(_f2, _f2))
	else:
		raise ValueError("weight is not supported")
		return np.dot(weight*_f1, _f2) / np.sqrt(np.dot(weight*_f1, weight*_f1) * np.dot(weight*_f2, weight*_f2))

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
	
