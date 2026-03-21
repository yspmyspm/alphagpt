import os
import torch


class ModelConfig:
    """
    训练/数据的全局配置。
    - 数据来自 rl-mining/olhcv.feather，不再依赖数据库。
    - 特征维度需与 FeatureEngineer 保持一致。
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # 本地 feather 数据路径
    FEATHER_PATH = os.getenv("FEATHER_PATH", os.path.join("data", "olhcv.feather"))

    # returns 来源：从 returns 文件读取（默认 returns/30min.feather）
    RETURNS_DIR = os.getenv("RETURNS_DIR", "returns")
    RETURNS_FILENAME = os.getenv("RETURNS_FILENAME", "30min.feather")
    RETURNS_COLUMN = "returns"  # returns 文件中的收益列名


    # 特征列：None 表示使用 feather 中所有数值列；也可指定列名列表
    FEATURE_COLUMNS = None

    # 训练超参
    BATCH_SIZE = 1024
    TRAIN_STEPS = 1000
    MAX_FORMULA_LEN = 12
    GEN_TEMPERATURE = 1.0
    TS_PARAMETERS = [1, 5, 10, 30, 60, 120, 1440]

    # Reward 平滑：True 时约束失败按合规程度线性插值，而非直接 -5
    USE_SMOOTH_REWARD = True
    BACKTEST_PENALTY = -5.0
    UNFINISHED_PENALTY = -5.0
    ICIR_MISSING_GAMMA = 2.0
    ICIR_MISSING_EPS = 1e-6

    # 并行评估：验证公式时使用的进程数，0 表示不并行
    EVAL_NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)

    # 特征维度：由 data_loader 加载后设置，与 feather 中特征列数一致
    INPUT_DIM = None
