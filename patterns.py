class CandlestickPatterns:

    @staticmethod
    def is_bullish_engulfing(candles):
        if len(candles) < 2:
            return False
        prev = candles[-2]
        curr = candles[-1]
        prev_bearish = prev['close'] < prev['open']
        curr_bullish = curr['close'] > curr['open']
        engulfs = curr['open'] <= prev['close'] and curr['close'] >= prev['open']
        return prev_bearish and curr_bullish and engulfs

    @staticmethod
    def is_bearish_engulfing(candles):
        if len(candles) < 2:
            return False
        prev = candles[-2]
        curr = candles[-1]
        prev_bullish = prev['close'] > prev['open']
        curr_bearish = curr['close'] < curr['open']
        engulfs = curr['open'] >= prev['close'] and curr['close'] <= prev['open']
        return prev_bullish and curr_bearish and engulfs

    @staticmethod
    def is_hammer(candle):
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        if total_range == 0:
            return False
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        return lower_shadow > body * 2 and upper_shadow < body * 0.5 and body / total_range < 0.3

    @staticmethod
    def is_shooting_star(candle):
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        if total_range == 0:
            return False
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        return upper_shadow > body * 2 and lower_shadow < body * 0.5 and body / total_range < 0.3

    @staticmethod
    def is_morning_star(candles):
        if len(candles) < 3:
            return False
        first = candles[-3]
        second = candles[-2]
        third = candles[-1]
        first_bearish = first['close'] < first['open']
        second_small = abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3
        third_bullish = third['close'] > third['open']
        return first_bearish and second_small and third_bullish and third['close'] > first['open']

    @staticmethod
    def is_evening_star(candles):
        if len(candles) < 3:
            return False
        first = candles[-3]
        second = candles[-2]
        third = candles[-1]
        first_bullish = first['close'] > first['open']
        second_small = abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3
        third_bearish = third['close'] < third['open']
        return first_bullish and second_small and third_bearish and third['close'] < first['open']