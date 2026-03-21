import copy
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(Path(__file__).parent))

from myops import *
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from toolkit import toolkit_scorer

from config.ts_para import *


class Feature:
	def __init__(self,data:pd.Series,postfix = None):
		self.data = data.copy()
		self.postfix = postfix
	
	def eval(self,**kwargs):
		returns = kwargs.get("returns")
		best_score = kwargs.get("best_score")
		best_postfix = kwargs.get("best_postfix")
		best_series = kwargs.get("best_series")
		result = toolkit_scorer(self.data,returns)
		if result is not None and abs(result['icir']) >abs(best_score):
			best_postfix = copy.copy(self.postfix)
			best_score = result['icir']
			best_series = self.data.copy()
		
		return best_postfix,best_series,best_score
		
class Parameter:
	def __init__(self, value: Any):
		self.value = value

class operators:
	# 这个类整合了所有的算子
	def __init__(self):
		self.binary = []
		for op_name in dir(binary):
			if callable(getattr(binary, op_name)) and not op_name.startswith("__"):
				self.binary.append(op_name)
				setattr(self, op_name, getattr(binary, op_name))
		
		self.unary_parameterized = []
		for op_name in dir(unary_parameterized):
			if callable(getattr(unary_parameterized, op_name)) and not op_name.startswith("__"):
				self.unary_parameterized.append(op_name)
				setattr(self, op_name, getattr(unary_parameterized, op_name))
		

		self.unary_parameterless = []
		for op_name in dir(unary_parameterless):
			if callable(getattr(unary_parameterless, op_name)) and not op_name.startswith("__"):
				self.unary_parameterless.append(op_name)
				setattr(self, op_name, getattr(unary_parameterless, op_name))
		

		self.ts_unary = []
		for op_name in dir(ts_unary):
			if callable(getattr(ts_unary, op_name)) and not op_name.startswith("__"):
				self.ts_unary.append(op_name)
				setattr(self, op_name, getattr(ts_unary, op_name))

		self.ts_binary = []
		for op_name in dir(ts_binary):
			if callable(getattr(ts_binary, op_name)) and not op_name.startswith("__"):
				self.ts_binary.append(op_name)
				setattr(self, op_name, getattr(ts_binary, op_name))

		self.ts_parameters = [str(p) for p in ts_para_list]
		self.operator_total = self.binary + self.unary_parameterized + self.unary_parameterless + self.ts_unary + self.ts_binary
		return


class TokenSet:
	def __init__(self,feature_list):
		self.feature_list = feature_list
		self.operators = operators()

class expression:
	def __init__(self,data):
		self.feature_list = data.columns.tolist()
		self.operators = operators()
		self.data = data.copy()
		return

	def evaluate_postfix(self,postfix: List[str]) -> pd.Series:
		"""计算解耦了参数的逆波兰表达式的值"""
		stack = []
		try:
			for token in postfix:
				if token in self.feature_list:
					stack.append(Feature(self.data[token]))
				elif token in self.operators.ts_parameters:
					stack.append(Parameter(int(token)))
				else:
					# It must be an operator
					op_name = token
					if not hasattr(self.operators, op_name):
						raise ValueError(f"Unknown operator: {op_name}")
					func = getattr(self.operators, op_name)
					
					
					# 二元运算符
					if op_name in self.operators.binary:
						if len(stack) < 2:
							raise ValueError(f"Not enough operands for binary operator: {op_name}")
						y = stack.pop()
						x = stack.pop()
						assert isinstance(x, Feature) and isinstance(y, Feature)
						result = func(x.data, y.data)

					# 有参数的二元运算符
					elif op_name in self.operators.ts_binary:
						if len(stack) < 3:
							raise ValueError(f"Not enough operands for binary operator: {op_name}")

						param = stack.pop()
						y = stack.pop()
						x = stack.pop()
						assert isinstance(x, Feature) and isinstance(y, Feature) and isinstance(param,Parameter)
						result = func(x.data, y.data,param.value)
					
					
					# 有参数的一元运算符
					elif op_name in self.operators.unary_parameterized or op_name in self.operators.ts_unary:
						if len(stack) < 2:
							raise ValueError(f"Not enough operands for parameterized unary operator: {op_name}")
						param = stack.pop()
						x = stack.pop()
						assert isinstance(x, Feature) and isinstance(param, Parameter)
						result = func(x.data, param.value)
					
					# 没有参数的一元运算符
					elif op_name in self.operators.unary_parameterless:
						if len(stack) < 1:
							raise ValueError(f"Not enough operands for unary operator: {op_name}")
						x = stack.pop()
						assert isinstance(x, Feature)
						result = func(x.data)

					stack.append(Feature(result))

		except Exception as e:
			print(f"Error evaluating postfix expression: {postfix}")
			print(f"Error details: {str(e)}")
			raise e
		
		
		return stack[0].data if stack and isinstance(stack[0], Feature) else pd.Series(index=self.data.index, dtype=np.float64)

		