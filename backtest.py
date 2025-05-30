import time
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import talib
import pandas as pd
import json
import os
from enum import Enum
from areixio import (
    BacktestBroker, CryptoDataFeed, create_report_folder,
    Strategy, BackTest, Statistic
)
# This script is designed to run a backtest for a trading strategy using the Areixio framework.
# It includes the use of technical indicators and candlestick patterns to generate buy/sell signals.
# To change the strategy, you can modify the `self.patterns` dictionary in the `initialize` method of the `TestStrategy` class.
# Make sure to adjust the `interval`, `tp_multiplier`, `sl_multiplier`, and other parameters as needed.
# For 4h strategy, set tp_multiplier to 0.05 and sl_multiplier to 0.025.
# For 6h strategy, set tp_multiplier to 0.055 and sl_multiplier to 0.035.
# For 1d strategy, set tp_multiplier to 0.1 and sl_multiplier to 0.08.

def run_backtest():
    class TestStrategy(Strategy):
        tp_multiplier = 0.05
        sl_multiplier = 0.025
        atr_period = 14
        sma_period = 30
        smma_period = 20
        tp_list = []
        sl_list = []

        def initialize(self):
            self.long_stop = defaultdict(float)
            self.long_take_profit = defaultdict(float)
            self.intra_trade_high = defaultdict(float)
            self.intra_trade_low = defaultdict(float)
            self.indicators = {}
            self.trade_signals = []
            self.detected_patterns = defaultdict(str)
            self.patterns = {
                "CDL2CROWS": talib.CDL2CROWS,
                "CDL3BLACKCROWS": talib.CDL3BLACKCROWS,
                "CDL3INSIDE": talib.CDL3INSIDE,
                "CDL3LINESTRIKE": talib.CDL3LINESTRIKE,
                "CDL3OUTSIDE": talib.CDL3OUTSIDE,
                "CDL3STARSINSOUTH": talib.CDL3STARSINSOUTH,
                "CDL3WHITESOLDIERS": talib.CDL3WHITESOLDIERS,
                "CDLABANDONEDBABY": lambda o, h, l, c: talib.CDLABANDONEDBABY(o, h, l, c, penetration=0),
                "CDLADVANCEBLOCK": talib.CDLADVANCEBLOCK,
                "CDLBELTHOLD": talib.CDLBELTHOLD,
                "CDLBREAKAWAY": talib.CDLBREAKAWAY,
                "CDLCLOSINGMARUBOZU": talib.CDLCLOSINGMARUBOZU,
                "CDLCONCEALBABYSWALL": talib.CDLCONCEALBABYSWALL,
                "CDLCOUNTERATTACK": talib.CDLCOUNTERATTACK,
                "CDLDARKCLOUDCOVER": lambda o, h, l, c: talib.CDLDARKCLOUDCOVER(o, h, l, c, penetration=0),
                "CDLDOJI": talib.CDLDOJI,
                "CDLDOJISTAR": talib.CDLDOJISTAR,
                "CDLDRAGONFLYDOJI": talib.CDLDRAGONFLYDOJI,
                "CDLENGULFING": talib.CDLENGULFING,
                "CDLEVENINGDOJISTAR": lambda o, h, l, c: talib.CDLEVENINGDOJISTAR(o, h, l, c, penetration=0),
                "CDLEVENINGSTAR": lambda o, h, l, c: talib.CDLEVENINGSTAR(o, h, l, c, penetration=0),
                "CDLGAPSIDESIDEWHITE": talib.CDLGAPSIDESIDEWHITE,
                "CDLGRAVESTONEDOJI": talib.CDLGRAVESTONEDOJI,
                "CDLHAMMER": talib.CDLHAMMER,
                "CDLHANGINGMAN": talib.CDLHANGINGMAN,
                "CDLHARAMI": talib.CDLHARAMI,
                "CDLHARAMICROSS": talib.CDLHARAMICROSS,
                "CDLHIGHWAVE": talib.CDLHIGHWAVE,
                "CDLHIKKAKE": talib.CDLHIKKAKE,
                "CDLHIKKAKEMOD": talib.CDLHIKKAKEMOD,
                "CDLHOMINGPIGEON": talib.CDLHOMINGPIGEON,
                "CDLIDENTICAL3CROWS": talib.CDLIDENTICAL3CROWS,
                "CDLINNECK": talib.CDLINNECK,
                "CDLINVERTEDHAMMER": talib.CDLINVERTEDHAMMER,
                "CDLKICKING": talib.CDLKICKING,
                "CDLKICKINGBYLENGTH": talib.CDLKICKINGBYLENGTH,
                "CDLLADDERBOTTOM": talib.CDLLADDERBOTTOM,
                "CDLLONGLEGGEDDOJI": talib.CDLLONGLEGGEDDOJI,
                "CDLLONGLINE": talib.CDLLONGLINE,
                "CDLMARUBOZU": talib.CDLMARUBOZU,
                "CDLMATCHINGLOW": talib.CDLMATCHINGLOW,
                "CDLMATHOLD": lambda o, h, l, c: talib.CDLMATHOLD(o, h, l, c, penetration=0),
                "CDLMORNINGDOJISTAR": lambda o, h, l, c: talib.CDLMORNINGDOJISTAR(o, h, l, c, penetration=0),
                "CDLMORNINGSTAR": lambda o, h, l, c: talib.CDLMORNINGSTAR(o, h, l, c, penetration=0),
                "CDLONNECK": talib.CDLONNECK,
                "CDLPIERCING": talib.CDLPIERCING,
                "CDLRICKSHAWMAN": talib.CDLRICKSHAWMAN,
                "CDLRISEFALL3METHODS": talib.CDLRISEFALL3METHODS,
                "CDLSEPARATINGLINES": talib.CDLSEPARATINGLINES,
                "CDLSHOOTINGSTAR": talib.CDLSHOOTINGSTAR,
                "CDLSHORTLINE": talib.CDLSHORTLINE,
                "CDLSPINNINGTOP": talib.CDLSPINNINGTOP,
                "CDLSTALLEDPATTERN": talib.CDLSTALLEDPATTERN,
                "CDLSTICKSANDWICH": talib.CDLSTICKSANDWICH,
                "CDLTAKURI": talib.CDLTAKURI,
                "CDLTASUKIGAP": talib.CDLTASUKIGAP,
                "CDLTHRUSTING": talib.CDLTHRUSTING,
                "CDLTRISTAR": talib.CDLTRISTAR,
                "CDLUNIQUE3RIVER": talib.CDLUNIQUE3RIVER,
                "CDLUPSIDEGAP2CROWS": talib.CDLUPSIDEGAP2CROWS,
                "CDLXSIDEGAP3METHODS": talib.CDLXSIDEGAP3METHODS
            }

            #Strategy 4h
            '''
            self.patterns = {
                'CDLSEPARATINGLINES': talib.CDLSEPARATINGLINES,
                'CDLHIKKAKEMOD': talib.CDLHIKKAKEMOD,
            }
            '''
            #Strategy 6h
            '''
            self.patterns = {
                'CDLHIKKAKEMOD': talib.CDLHIKKAKEMOD,
                'CDLDRAGONFLYDOJI': talib.CDLDRAGONFLYDOJI,
                'CDLTAKURI': talib.CDLTAKURI
            }
            '''
            

            #Strategy 1d
            '''
            self.patterns = {
                "CDLHAMMER": talib.CDLHAMMER,
                "Hikkake": talib.CDLHIKKAKE,
                "CDLTAKURI": talib.CDLTAKURI
            }
            '''

            for code, exchange in self.ctx.symbols:
                self.indicators[code] = []

        def on_order_fill(self, order):
            self.info(f"Order {order['order_id']} filled at {order['price']}")
            
        def serialize_signal(self):
            """Serialize trade signals to JSON."""
            def default_serializer(obj):
                if isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.strftime('%Y-%m-%d %H:%M:%S%z')
                elif isinstance(obj, Enum):
                    return obj.value
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"Unserializable Type: {type(obj)}")

            target_dir = r"D:\jupyter notebook\Program" # Target directory
            os.makedirs(target_dir, exist_ok=True)
            file_path = os.path.join(target_dir, 'trade_signals.json')

            try:
                with open(file_path, 'w') as f:
                    json.dump(
                        self.trade_signals,
                        f,
                        default=default_serializer,
                        indent=4
                    )
                #print(f"Signals saved to: {os.path.abspath(file_path)}")
            except Exception as e:
                print(f"Saving failed: {str(e)}")

        def on_bar(self, tick):
            self.cancel_all()

            for code, exchange in self.ctx.symbols:
                bar = self.ctx.get_bar_data(code, exchange)
                if bar is None:
                    continue

                self.indicators[code].append(bar)
                if len(self.indicators[code]) < self.sma_period:
                    continue

                close = np.array([b.close for b in self.indicators[code]])
                open = np.array([b.open for b in self.indicators[code]])
                high = np.array([b.high for b in self.indicators[code]])
                low = np.array([b.low for b in self.indicators[code]])
                volume = np.array([b.volume for b in self.indicators[code]])

                
                stddev = talib.STDDEV(close, timeperiod=14)[-1]
                sma = talib.SMA(close, timeperiod=self.sma_period)[-1]
                smma = talib.EMA(close, timeperiod=self.smma_period)[-1]
                atr = talib.ATR(high, low, close, timeperiod=self.atr_period)[-1]
                atr_upper = sma + atr
                #atr_lower = sma - atr
                volatility = stddev
                current_price = bar.close

                pattern_detected = False
                for name, func in self.patterns.items():
                    value = func(open, high, low, close)
                    if value[-1] != 0:
                        self.detected_patterns[code] = name
                        pattern_detected = True
                        break

                break_above_atr = current_price > atr_upper

                position = self.ctx.get_quantity(code, exchange)
                order = None
                sell_order = None
                

                if pattern_detected and break_above_atr:
                    entry_price = current_price
                    tp_atr_n = 2  # 止盈为2倍ATR
                    sl_atr_n = 1  # 止损为1倍ATR
                    take_profit = entry_price + tp_atr_n * atr
                    stop_loss = entry_price - sl_atr_n * atr
                    

                    if entry_price * self.fixed_size[code] < self.available_balance:
                        order = self.buy(code, exchange, self.fixed_size[code])
                        self.tp_list.append(take_profit)
                        self.sl_list.append(stop_loss)
                        self.long_stop[code] = stop_loss
                        self.long_take_profit[code] = take_profit
                        self.info(f"Buy on pattern + ATR break: {order}")
                        if order:
                            self.trade_signals.append({
                                'Order ID': order['order_id'],
                                'Timestamp': bar.datetime,
                                'Code': code,
                                'Signal': 'Buy Long',
                                'Entry Price': current_price,
                                'Current position': position,
                                'Pattern': self.detected_patterns[code],
                                'Take Profit': take_profit,
                                'Stop Loss': stop_loss,
                                'Volume': float(bar.volume),
                                'Volatility': float(volatility),
                                'ATR': float(atr)
                            })

                if position > 0:
                    remaining = position

                    for tp in sorted(self.tp_list):
                        if current_price >= tp:
                            qty = min(remaining, self.fixed_size[code])
                            sell_order = self.sell(code, exchange, qty)
                            self.tp_list.remove(tp)
                            remaining -= qty
                            if remaining <= 0:
                                self.tp_list.clear()
                                self.sl_list.clear()
                                break

                    for sl in sorted(self.sl_list):
                        if current_price <= sl and remaining > 0:
                            qty = min(remaining, self.fixed_size[code])
                            sell_order = self.sell(code, exchange, qty)
                            self.sl_list.remove(sl)
                            remaining -= qty
                            if remaining <= 0:
                                self.tp_list.clear()
                                self.sl_list.clear()
                                break
                if len(self.trade_signals) > 0:  # Avoid empty signals
                    self.serialize_signal()

    currentDateAndTime = datetime.now()

    interval = '6h'
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    #end_date = currentDateAndTime.strftime("%Y-%m-%d") #Today
    fixed_size = {'BTCUSDT': 0.01}
    codes = list(fixed_size.keys())
    exchange = 'bybit'
    asset_type = 'perpetual'
    base = create_report_folder()

    feeds = [CryptoDataFeed(code=c, exchange=exchange, asset_type=asset_type,
                             start_date=start_date, end_date=end_date,
                             interval=interval, order_ascending=True) for c in codes]

    benchmark = CryptoDataFeed(code='BTCUSDT', exchange=exchange,
                                asset_type=asset_type, start_date=start_date,
                                end_date=end_date, interval=interval)

    broker = BacktestBroker(balance=1000000, slippage=0.0)
    statistic = Statistic()

    mytest = BackTest(feeds, TestStrategy, statistic=statistic,
                      benchmark=benchmark, store_path=base, broker=broker,
                      backtest_mode='bar', fixed_size=fixed_size)

    mytest.start()
    stats = mytest.ctx.statistic.stats(interval=interval)
    stats['algorithm'] = ['Pattern + ATR Break']
    print(stats)
    mytest.ctx.statistic.contest_output(path=base, interval=interval, prefix='bt_', is_plot=True)

if __name__ == '__main__':
    #Change to while True if keep running needed
    if True:
        run_backtest()
        #time.sleep(60)  # Wait for 60 seconds before the next run