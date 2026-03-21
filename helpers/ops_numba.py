from numba import njit
import numpy as np

class ts_numba:
	@njit
	def ts_skew(x, window):
		n = len(x)
		out = np.full(n, np.nan)
		coef = window*window / (window-1)/(window-2)
		for i in range(window-1, n):
			y = x[i-window+1:i+1]
			std = np.nanstd(y)
			mean = np.nanmean(y)
			out[i] = np.nanmean(((y - mean)/(std * np.sqrt(window/(window-1))))**3)
			out[i] *= coef
		return out	

	@njit
	def ts_argmax(x,window):
		n = len(x)
		out = np.full(n, np.nan)
		for i in range(window-1,n):
			out[i] = np.argmax(x[i-window+1:i+1])
		return out+1

	@njit
	def ts_argmin(x,window):
		n = len(x)
		out = np.full(n, np.nan)
		for i in range(window-1,n):
			out[i] = np.argmin(x[i-window+1:i+1])
		return out+1

	@njit
	def ts_mad(x, window):
		n = len(x)
		out = np.full(n, np.nan)
		s1 = 0
		for i in range(window-1,n):
			out[i] = np.nanmean(np.abs(x[i-window+1:i+1] - np.nansum(x[i-window+1:i+1])/window))
		return out

	@njit
	def ts_kurt(x, window):
		n = len(x)
		out = np.full(n, np.nan)
		for i in range(window-1, n):
			y = x[i-window+1:i+1]
			std = np.nanstd(y)
			mean = np.nanmean(y)
			m4 = np.nansum(((y - mean)/(std * np.sqrt(window/(window-1))))**4)
			m2 = np.nansum(((y - mean)/(std * np.sqrt(window/(window-1))))**2)
			out[i] = m4 * (window+1) * window - m2*m2 * (window-1)*3
		return out/(window-1)/(window-2)/(window-3)

	@njit
	def ts_softmax(x, window):
		n = len(x)
		out = np.full(n, np.nan)
		for i in range(window-1, n):
			y = x[i-window+1:i+1].copy()
			y -= y.max()
			y = np.exp(y)
			nansum_y = np.nansum(y)
			if nansum_y == 0:
				continue
			out[i] = y[-1]/nansum_y
		return out	
	
	@njit
	def ts_rcor(x,y, window):
		n = len(x)
		out = np.full(n, np.nan)
		for i in range(window-1, n):
			_x = x[i-window+1:i+1]
			_y = y[i-window+1:i+1]
			x_mod = np.sqrt(np.nansum(_x**2))
			y_mod = np.sqrt(np.nansum(_y**2))
			x_mod = max(x_mod,1e-8)
			y_mod = max(y_mod,1e-8)
			out[i] = np.nansum(_x * _y) / x_mod / y_mod
		return out

	@njit
	def ts_rank(x, window):
		n = len(x)
		out = np.full(n, np.nan)
		for i in range(window - 1, n):
			y = x[i - window + 1 : i + 1]
			val = y[-1]
			count = np.nansum(y < val) + np.nansum(y == val) * 0.5
			out[i] = count / (window - 1) 
		return out

	@njit
	def ts_wmean(x, window):
		n = len(x)
		out = np.full(n, np.nan)
		weights = np.arange(1, window + 1, dtype=np.float64)
		w_sum = np.nansum(weights)

		for i in range(window - 1, n):
			y = x[i - window + 1 : i + 1]
			out[i] = np.nansum(y * weights) / w_sum
		return out

	@njit
	def ts_expwmean(x, window):
		n = len(x)
		out = np.full(n, np.nan)
		weights = np.arange(1, window + 1, dtype=np.float64)
		weights = np.exp(-weights)
		w_sum = np.nansum(weights)

		for i in range(window - 1, n):
			y = x[i - window + 1 : i + 1]
			out[i] = np.nansum(y * weights) / w_sum
		return out

	@njit
	def ts_reg_beta(x, window):
		n = len(x)
		out = np.full(n, np.nan)
		t = np.arange(window, dtype=np.float64)
		t_mean = np.nanmean(t)
		t_var = np.nanmean((t - t_mean)**2)
		for i in range(window - 1, n):
			y = x[i - window + 1 : i + 1]
			y_mean = np.nanmean(y)
			numerator = np.nanmean((y - y_mean) * (t - t_mean))
			out[i] = numerator / t_var
		return out

	@njit
	def ts_reg_resid(x, window):
		n = len(x)
		out = np.full(n, np.nan)

		t = np.arange(window, dtype=np.float64)
		t_mean = np.nanmean(t)
		t_var = np.nanmean((t - t_mean)**2)
		t_last = float(window - 1)

		for i in range(window - 1, n):
			y = x[i - window + 1 : i + 1]
			y_mean = np.nanmean(y)

			numerator = np.nanmean((y - y_mean) * (t - t_mean))
			beta = numerator / t_var
			alpha = y_mean - beta * t_mean

			# 最后一个点的实际值 y[-1] 减去 预测值
			out[i] = y[-1] - (alpha + beta * t_last)
		return out



