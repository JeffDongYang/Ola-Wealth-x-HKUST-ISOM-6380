import gym
from gym import spaces
import numpy as np
import json

import gym
from gym import spaces
import numpy as np
from stable_baselines3 import SAC

class TradingEnv(gym.Env):
    def __init__(self, data, tp_range=(0.01,0.3), sl_range=(0.01,0.1)):
        super().__init__()
        self.data = data
        self.tp_range = tp_range
        self.sl_range = sl_range
        self.current_step = 0

        # 构建信号和pattern的one-hot空间
        self.signal_types = sorted(list(set(d['Signal'] for d in data)))
        self.patterns = sorted(list(set(d['Pattern'] for d in data)))
        self.codes = sorted(list(set(d['Code'] for d in data)))
        
        self.n_signals = len(self.signal_types)
        self.n_patterns = len(self.patterns)
        self.n_codes = len(self.codes)
        
        self.numeric_features = 8 # [volume, volatility, tp, sl, hour_of_day, day_of_week, ATR]
        obs_len = self.n_signals + self.n_patterns + self.n_codes + self.numeric_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_signals + self.n_patterns + self.n_codes + self.numeric_features,),
            dtype=np.float32
        )
        # 连续动作空间 [tp, sl]
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
        signal_onehot = np.zeros(self.n_signals, dtype=np.float32)
        pattern_onehot = np.zeros(self.n_patterns, dtype=np.float32)
        code_onehot = np.zeros(self.n_codes, dtype=np.float32)
        signal_onehot[self.signal_types.index(d['Signal'])] = 1
        pattern_onehot[self.patterns.index(d['Pattern'])] = 1
        code_onehot[self.codes.index(d['Code'])] = 1
    
        volume = float(d.get('Volume', 0))
        volatility = float(d.get('Volatility', 0))
        tp = float(d.get('Take Profit', 0))
        sl = float(d.get('Stop Loss', 0))
        entry_price = float(d.get('Entry Price', 0))
        atr = float(d.get('ATR', 0))
    
        # 时间特征
        try:
            dt = pd.to_datetime(d['Timestamp'])
            hour_of_day = dt.hour / 23.0
            day_of_week = dt.weekday() / 6.0
        except Exception:
            hour_of_day = 0.0
            day_of_week = 0.0
    
        obs = np.concatenate([
            signal_onehot, pattern_onehot, code_onehot,
            [volume, volatility, tp, sl, hour_of_day, day_of_week, entry_price, atr]
        ])
        if obs.shape[0] != self.observation_space.shape[0]:
            raise ValueError(f"obs shape {obs.shape} != {self.observation_space.shape}")
        return obs

    def step(self, action):
        tp, sl = action
        d = self.data[self.current_step]
        best_tp, best_sl = d['Take Profit'], d['Stop Loss']
    
        reward = - (abs(tp - best_tp) / (abs(best_tp)+1e-8) + abs(sl - best_sl) / (abs(best_sl)+1e-8))
    
        self.current_step += 1
        done = self.current_step >= len(self.data)
        obs_dim = self.observation_space.shape[0]
        obs = self._get_obs() if not done else np.zeros(obs_dim, dtype=np.float32)
        info = {}
        return obs, reward, done, info

    def render(self, mode='human'):
        pass  # 可扩展为打印当前step等

    def close(self):
        pass


with open("trade_signals.json", "r", encoding="utf-8") as f:
    signals_data = json.load(f)

filtered_signals = [
    d for d in signals_data
    if 'Signal' in d and 'Pattern' in d and 'Take Profit' in d and 'Stop Loss' in d
]
print(f"原始信号数量: {len(signals_data)}，过滤后信号数量: {len(filtered_signals)}")

env = TradingEnv(filtered_signals)
model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100_000)  # 你可以根据数据量调整
model.save("sac_trading_agent")
obs = env.reset()
done = False
total_reward = 0
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
print(f"总奖励: {total_reward}")