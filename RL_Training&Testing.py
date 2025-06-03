import gym
from gym import spaces
import numpy as np
import pandas as pd
import pickle
import json
from stable_baselines3 import SAC

# ========= 1. 数据预处理 =========
with open("trade_signals.json", "r", encoding="utf-8") as f:
    signals_data = json.load(f)

filtered_signals = [
    d for d in signals_data
    if 'Signal' in d and 'Pattern' in d and 'Take Profit' in d and 'Stop Loss' in d
]

# 统计所有出现过的类别
all_signals = sorted(list(set(d['Signal'] for d in filtered_signals)))
all_patterns = sorted(list(set(d['Pattern'] for d in filtered_signals)))
all_codes = sorted(list(set(d['Code'] for d in filtered_signals)))

# 增加"other"类别，必须放在最后一位
all_signals.append('other')
all_patterns.append('other')
all_codes.append('other')

# 保存类别，推理时要用
with open("env_categories.pkl", "wb") as f:
    pickle.dump({
        "signal_types": all_signals,
        "patterns": all_patterns,
        "codes": all_codes
    }, f)

# ========= 2. 环境定义 =========
def get_index_with_other(cat_list, item):
    try:
        return cat_list.index(item)
    except ValueError:
        return len(cat_list) - 1  # "other"

class TradingEnv(gym.Env):
    def __init__(self, data, signal_types, patterns, codes,
                 tp_range=(0.9, 1.25), sl_range=(0.95, 1.01)):
        super().__init__()
        self.data = data
        self.signal_types = signal_types
        self.patterns = patterns
        self.codes = codes
        self.n_signals = len(signal_types)
        self.n_patterns = len(patterns)
        self.n_codes = len(codes)
        self.numeric_features = 8
        self.current_step = 0

        obs_len = self.n_signals + self.n_patterns + self.n_codes + self.numeric_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_len,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([tp_range[0], sl_range[0]], dtype=np.float32),
            high=np.array([tp_range[1], sl_range[1]], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        d = self.data[self.current_step]
        # one-hot，未知类别归到"other"位
        signal_onehot = np.zeros(self.n_signals, dtype=np.float32)
        pattern_onehot = np.zeros(self.n_patterns, dtype=np.float32)
        code_onehot = np.zeros(self.n_codes, dtype=np.float32)
        si = get_index_with_other(self.signal_types, d['Signal'])
        pi = get_index_with_other(self.patterns, d['Pattern'])
        ci = get_index_with_other(self.codes, d['Code'])
        signal_onehot[si] = 1
        pattern_onehot[pi] = 1
        code_onehot[ci] = 1

        entry_price = float(d.get('Entry Price', 1))
        volume = float(d.get('Volume', 0)) / entry_price
        volatility = float(d.get('Volatility', 0)) / entry_price
        tp = float(d.get('Take Profit', 0)) / entry_price
        sl = float(d.get('Stop Loss', 0)) / entry_price
        atr = float(d.get('ATR', 0)) / entry_price

        try:
            dt = pd.to_datetime(d['Timestamp'])
            hour_of_day = dt.hour / 23.0
            day_of_week = dt.weekday() / 6.0
        except Exception:
            hour_of_day = 0.0
            day_of_week = 0.0

        obs = np.concatenate([
            signal_onehot, pattern_onehot, code_onehot,
            [volume, volatility, tp, sl, hour_of_day, day_of_week, 1.0, atr]
        ])
        if obs.shape[0] != self.observation_space.shape[0]:
            raise ValueError(f"obs shape {obs.shape} != {self.observation_space.shape}")
        return obs

    def step(self, action):
        tp, sl = action
        d = self.data[self.current_step]
        entry_price = float(d.get('Entry Price', 1))
        best_tp = float(d['Take Profit']) / entry_price
        best_sl = float(d['Stop Loss']) / entry_price
        diff = abs(tp - best_tp) + abs(sl - best_sl)
        reward = 1.0 if diff < 0.01 else -1.0
        self.current_step += 1
        done = self.current_step >= len(self.data)
        obs_dim = self.observation_space.shape[0]
        obs = self._get_obs() if not done else np.zeros(obs_dim, dtype=np.float32)
        info = {}
        return obs, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# ========= 3. 训练 =========
env = TradingEnv(
    filtered_signals,
    signal_types=all_signals,
    patterns=all_patterns,
    codes=all_codes
)
model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("sac_trading_agent")