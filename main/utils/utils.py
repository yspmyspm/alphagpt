import os
import json
import re
 
def normalize_optional_path(path):
	if path is None:
		return None
	s = str(path).strip()
	if not s:
		return None
	return os.path.abspath(os.path.expanduser(s))

def json_compact_number_lists(obj) -> str:
	"""Serialize obj to JSON (indent=2) but render lists of numbers on a single line."""
	raw = json.dumps(obj, ensure_ascii=False, indent=2)
	_num_pat = re.compile(
		r'\[\s*\n\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
		r'(?:\s*,\s*\n\s*-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)*'
		r'\s*\n\s*\]'
	)
	def _collapse(m: re.Match) -> str:
		return re.sub(r'\s+', ' ', m.group(0)).replace('[ ', '[').replace(' ]', ']')
	return _num_pat.sub(_collapse, raw)

def _format_token_line(tokens) -> str:
	return ", ".join(str(token) for token in tokens) if tokens else "(none)"
