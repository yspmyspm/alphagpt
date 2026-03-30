import pandas as pd
import numpy as np

from config import ModelConfig

from .ops_numba import ts_numba


class ts_unary:
	@staticmethod
	def ts_lag(df: pd.Series, t: int = 1) -> pd.Series:
		return df.shift(t)

	@staticmethod
	def ts_delta(df:pd.Series, period=1):
		return df.diff(period)

	@staticmethod
	def ts_change_ratio(x:pd.Series,window=10)-> pd.Series:
		return x.diff(window) / x.shift(window)

	@staticmethod
	def ts_sum(df: pd.Series, window: int = 10) -> pd.Series:
		return df.rolling(window,min_periods=1).sum()

	@staticmethod
	def ts_mean(df:pd.Series, window=10)->pd.Series:
		return df.rolling(window,min_periods=1).mean()

	@staticmethod
	def ts_std(df:pd.Series, window=10)->pd.Series:
		return df.rolling(window,min_periods=1).std()

	@staticmethod
	def ts_min(df:pd.Series, window=10)->pd.Series:
		return df.rolling(window,min_periods=1).min()

	@staticmethod
	def ts_max(df:pd.Series, window=10)->pd.Series:
		return df.rolling(window,min_periods=1).max()

	@staticmethod
	def ts_range(x:pd.Series, window=10)->pd.Series:
		return x.rolling(window,min_periods=1).max() - x.rolling(window,min_periods=1).min()

	@staticmethod
	def ts_norm(x:pd.Series, window=10)->pd.Series:
		Min = x.rolling(window,min_periods=1).min()
		Max = x.rolling(window,min_periods=1).max()
		return (x - Min)/(Max-Min)

	@staticmethod
	def ts_median(x:pd.Series, window=10)->pd.Series:
		return x.rolling(window,min_periods=1).median()

	@staticmethod
	def ts_skew(x:pd.Series,window = 10) ->pd.Series:
		return pd.Series(ts_numba.ts_skew(x.to_numpy(),window),index=x.index)

	@staticmethod
	def ts_kurt(x:pd.Series,window = 10) ->pd.Series:
		return pd.Series(ts_numba.ts_kurt(x.to_numpy(),window),index=x.index)

	@staticmethod
	def ts_argmin(x:pd.Series,window = 10) ->pd.Series:
		return pd.Series(ts_numba.ts_argmin(x.to_numpy(),window),index=x.index)

	@staticmethod
	def ts_argmax(x:pd.Series,window = 10) ->pd.Series:
		return pd.Series(ts_numba.ts_argmax(x.to_numpy(),window),index=x.index)

	@staticmethod
	def ts_mad(x:pd.Series,window = 10) ->pd.Series:
		return pd.Series(ts_numba.ts_mad(x.to_numpy(),window),index=x.index)

	@staticmethod
	def ts_zscore(x: pd.Series, window=10)->pd.Series:
		mean = x.rolling(window,min_periods=1).mean()
		std = x.rolling(window,min_periods=1).std()
		return (x - mean) / std

	@staticmethod
	def ts_demean(x: pd.Series, window=10)->pd.Series:
		mean = x.rolling(window,min_periods=1).mean()
		return x - mean

	@staticmethod
	def ts_softmax(x:pd.Series,window = 10) ->pd.Series:
		return pd.Series(ts_numba.ts_softmax(x.to_numpy(),window),index=x.index)
	

	@staticmethod
	def ts_autocorr(x:pd.Series,window = 10) -> pd.Series:
		y = x.shift(1)
		return x.rolling(window,min_periods=1).corr(y)

	@staticmethod
	def ts_rank(x: pd.Series, window=10) -> pd.Series:
		"""Rolling Percentile Rank"""
		return pd.Series(ts_numba.ts_rank(x.to_numpy(), window), index=x.index)

	@staticmethod
	def ts_wmean(x: pd.Series, window=10) -> pd.Series:
		"""Linear Decay Weighted Mean"""
		return pd.Series(ts_numba.ts_wmean(x.to_numpy(), window), index=x.index)

	@staticmethod
	def ts_beta(x: pd.Series, window=10) -> pd.Series:
		"""Rolling Regression Slope (Slope of x against time)"""
		return pd.Series(ts_numba.ts_reg_beta(x.to_numpy(), window), index=x.index)

	@staticmethod
	def ts_resid(x: pd.Series, window=10) -> pd.Series:
		"""Rolling Regression Residuals"""
		return pd.Series(ts_numba.ts_reg_resid(x.to_numpy(), window), index=x.index)

	@staticmethod
	def ts_std_dev_diff(x: pd.Series, window=10) -> pd.Series:
		"""Std Dev Difference: 当前 Std 与前一期 Std 的差分 (用于衡量波动率变化)"""
		# 这种可以通过组合现有算子实现，不需要 Numba
		s = x.rolling(window,min_periods=1).std()
		return s.diff()

class unary_parameterless:
	@staticmethod
	def Sign(x: pd.Series):
		return np.sign(x)

	@staticmethod
	def Sqrt(x: pd.Series):
		return np.sqrt(np.abs(x))

	@staticmethod
	def Signed_sqrt(x: pd.Series):
		return np.sign(x) * np.sqrt(np.abs(x))

	@staticmethod
	def Log1p(x: pd.Series):
		x = x.abs()
		return np.log1p(x)

	@staticmethod
	def sin(x: pd.Series):
		x.replace([np.inf, -np.inf], np.nan, inplace=True)
		return np.sin(x)

	@staticmethod
	def cos(x: pd.Series):
		x.replace([np.inf, -np.inf], np.nan, inplace=True)
		return np.cos(x)

	@staticmethod
	def sigmoid(x: pd.Series):
		return 1.0 / (1.0 + np.exp((-x).clip(max = 100)))

	@staticmethod
	def softsign(x: pd.Series):
		return x / (1.0 + np.abs(x))

	@staticmethod
	def Log(x:pd.Series):
		temp = np.log(np.abs(x) + 1e-8)
		return pd.Series(temp, index=x.index, name=x.name)

	@staticmethod
	def Abs(x:pd.Series):
		return abs(x)

	@staticmethod
	def Inv(x:pd.Series):
		return 1 / unary_parameterless.safe_den(x)

	@staticmethod
	def Neg(x:pd.Series):
		return -x

	@staticmethod
	def tanh(x:pd.Series):
		temp = np.tanh(x)
		return pd.Series(temp, index=x.index, name=x.name)

	@staticmethod
	def Relu(x:pd.Series):
		temp =  np.maximum(0, x)
		return pd.Series(temp, index=x.index, name=x.name)


	@staticmethod
	def Leaky_relu(x:pd.Series)->pd.Series:
		temp = np.where(x > 0, x, 0.01 * x)
		return pd.Series(temp,index = x.index, name=x.name)

	@staticmethod
	def Power2(x:pd.Series):
		return x ** 2

	@staticmethod
	def Power3(x:pd.Series):
		return x**3

	@staticmethod
	def Power15(x:pd.Series):
		return x**1.5

	@staticmethod
	def safe_den(y):
		y = y.copy()
		y[abs(y)<1e-8] = np.sign(y[abs(y)<1e-8]) *1e-8
		return y

class unary_parameterized:
	pass

class binary:
	@staticmethod
	def Add(x, y):
		return x + y

	@staticmethod
	def Subtract(x, y):
		return x - y

	@staticmethod
	def Multiply(x, y):
		return x * y

	@staticmethod
	def Divide(x, y):
		return x / unary_parameterless.safe_den(y)

	@staticmethod
	def Min(x, y):
		return x.where(x < y, y)

	@staticmethod
	def Max(x, y):
		return x.where(x > y, y)

	@staticmethod
	def Mean(x,y):
		return (x + y) / 2

	@staticmethod
	def AbsDiff(x,y):
		return (x-y).abs()

	@staticmethod
	def RelDiff(x, y):
		return binary.Divide(x-y,y.abs())

	@staticmethod
	def LogRatio(x, y, eps = 1e-8):
		return np.log((x / unary_parameterless.safe_den(y)).abs() + eps)

	@staticmethod
	def Greater(x, y):
		return (x >  y).astype(int)

	@staticmethod
	def Less(x, y):
		return (x <  y).astype(int)

	@staticmethod
	def GreaterEqual(x, y):
		return (x >= y).astype(int)

	@staticmethod
	def LessEqual(x, y):
		return (x <= y).astype(int)

	@staticmethod
	def Equal(x, y):
		return (x == y).astype(int)

	@staticmethod
	def NotEqual(x, y):
		return (x != y).astype(int)

class ts_binary:
    @staticmethod
    def ts_corr(x: pd.Series, y: pd.Series, window) -> pd.Series:
        return x.rolling(window, min_periods=1).corr(y)

    @staticmethod
    def ts_cov(x: pd.Series, y: pd.Series, window) -> pd.Series:
        return x.rolling(window, min_periods=1).cov(y)

    @staticmethod
    def ts_rcor(x: pd.Series, y: pd.Series, window=10) -> pd.Series:
        return pd.Series(ts_numba.ts_rcor(x.to_numpy(), y.to_numpy(), window), index=x.index)

    @staticmethod
    def ts_beta_binary(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        cov_xy = ts_binary.ts_cov(x, y, window)
        var_y = y.rolling(window, min_periods=1).var()
        var_y = unary_parameterless.safe_den(var_y)
        return cov_xy / var_y

    @staticmethod
    def ts_reg_resid_binary(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        roll_x = x.rolling(window, min_periods=1)
        roll_y = y.rolling(window, min_periods=1)

        mean_x = roll_x.mean()
        mean_y = roll_y.mean()

        beta = ts_binary.ts_beta_binary(x, y, window)
        alpha = mean_x - beta * mean_y

        return x - (alpha + beta * y)

    @staticmethod
    def ts_spread_zscore(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        spread = x - y
        return ts_unary.ts_zscore(spread, window)

    @staticmethod
    def ts_logratio_zscore(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        logratio = binary.LogRatio(x, y)
        return ts_unary.ts_zscore(logratio, window)

    @staticmethod
    def ts_absdiff_mean(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        return (x - y).abs().rolling(window, min_periods=1).mean()

    @staticmethod
    def ts_squared_diff_mean(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        return ((x - y) ** 2).rolling(window, min_periods=1).mean()


# --- 算子注册（原 helpers/ops.py）：从下方类收集 (name, func, kind, arity)，供词表 / 约束解码 / VM 共用 ---


def _collect_class_ops(cls, kind, arity):
    out = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        attr = getattr(cls, name)
        if not callable(attr):
            continue
        out.append((name, attr, kind, arity))
    return out


def get_op_specs():
    """
    返回 [(name, func, kind, arity), ...]
    kind:
      - binary: 2 个 Feature
      - unary_parameterless: 1 个 Feature
      - unary_parameterized: 1 个 Feature + 1 个 Parameter
      - ts_unary: 1 个 Feature + 1 个 Parameter
      - ts_binary: 2 个 Feature + 1 个 Parameter
    arity 为栈弹出元素数量（含参数）。
    """
    specs = []
    specs.extend(_collect_class_ops(binary, "binary", 2))
    specs.extend(_collect_class_ops(unary_parameterized, "unary_parameterized", 2))
    specs.extend(_collect_class_ops(unary_parameterless, "unary_parameterless", 1))
    specs.extend(_collect_class_ops(ts_unary, "ts_unary", 2))
    specs.extend(_collect_class_ops(ts_binary, "ts_binary", 3))
    return specs


def get_ops():
    """兼容旧接口：返回 [(name, func, arity), ...]。"""
    return [(name, func, arity) for name, func, _, arity in get_op_specs()]


def get_ts_parameters():
    return [int(v) for v in ModelConfig.TS_PARAMETERS]


"""
# Ts_rank_amean
# Ts_rank_gmean
# Ts_rank_gmean_amean_diff
# Ts_ir
# Ts_mon
# Ts_max_to_min
# Ts_maxmin_norm
# Ts_to_max
# Ts_to_min
# Ts_min_max_cps
# Ts_min_max_diff
# Ts_to_mean
# Ts_to_ewm
# Ts_to_wmean
# Ts_pctchg_abs
# Ts_pctchg
# Ts_log_pctchg
# Ts_hhi
# Ts_rsi
# Ts_rankcorr
# Ts_cokurt
# Ts_coskew
# Ts_fxcut_75
# Ts_fxcut_50
# Ts_fxumr_75
# Ts_fxumr_50
# Ts_fxzscore_50
# Ts_fxzscore_75
# Ts_cut_mean
# Ts_cut_ewm
# Ts_umr_mean
# Ts_umr_ewm
# Ts_diffrankcorr
# Ts_diffcorr

# Curt
# Demean
# Scale
# Maxminnorm
# Sign
# Softmax_one
# S_log_1p
# Arc_tan
# Truncate
# Winsorize
# Quantile
# Rank_by_side
# Hhi
# Fraction
# Signed_power
# Cut
# Umr
# Ortho
# Regression_neut
# Regression_proj
# Poly_regression
# If_then_else
# Add_div
# Sub_add_div
# Sub_div

"""

    
