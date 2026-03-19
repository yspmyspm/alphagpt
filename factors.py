"""
FeatureEngineer：由 data_loader 设置 INPUT_DIM，供 VM / AlphaGPT 使用。
"""


class FeatureEngineer:
    INPUT_DIM = None  # 由 data_loader.load_data() 设置
