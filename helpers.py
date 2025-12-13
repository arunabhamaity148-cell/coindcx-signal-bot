"""
LOGIC EVALUATOR
Combines all 45 UniqueLogics into one decision engine
"""

from strategy.unique_logics import UniqueLogics
from strategy.helpers import bool_score, normalize_score

class LogicEvaluator(UniqueLogics):

    def evaluate_all_logics(
        self,
        df,
        orderbook,
        funding_rate,
        oi_history,
        recent_trades,
        fear_greed_index,
        news_times,
        liquidation_clusters
    ):
        score = 0
        reasons = []

        # ================= MARKET HEALTH =================
        btc_calm = self.check_btc_calm(df)
        score += bool_score(btc_calm, 10)

        regime = self.detect_market_regime(df)
        score += 10 if regime == 'trending' else 5

        funding_ok = self.check_funding_rate_normal(funding_rate)
        score += bool_score(funding_ok, 5)

        fear_ok = self.check_fear_greed_index(fear_greed_index)
        score += bool_score(fear_ok, 5)

        # ================= PRICE ACTION =================
        structure = self.detect_market_structure_shift(df)
        if structure == 'bullish':
            score += 10
        elif structure == 'bearish':
            score += 10

        atr = self.calculate_atr_filter(df)

        # ================= ORDERBOOK =================
        ob_imb = self.check_orderbook_imbalance(orderbook)
        if ob_imb['signal'] != 'neutral':
            score += 10

        vwap_dev = self.check_vwap_deviation(df)
        if vwap_dev['signal'] != 'normal':
            score += 5

        # ================= ANTI-TRAP =================
        cooldown_ok, losses = self.check_consecutive_losses(recent_trades)
        score += bool_score(cooldown_ok, 10)

        sl_hunt = self.detect_sl_hunting_zone(df)
        if sl_hunt:
            score -= 10

        # ================= FINAL =================
        final_score = normalize_score(score, 0, 100)

        trade_allowed = final_score >= 65

        return {
            'final_score': final_score,
            'trade_allowed': trade_allowed,
            'market_health': {
                'btc_calm': btc_calm,
                'market_regime': regime
            },
            'price_action': {
                'structure': structure,
                'atr': atr
            },
            'orderflow': {
                'orderbook_signal': ob_imb['signal']
            },
            'cooldown': {
                'allowed': cooldown_ok,
                'losses': losses
            }
        }