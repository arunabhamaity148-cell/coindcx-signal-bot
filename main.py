import time
import logging
from data.coindcx_connector import CoinDCXConnector
from strategy.unique_logics import LogicEvaluator
from strategy.institutional_scanner import InstitutionalScanner
from risk.risk_manager import RiskManager
from config.settings import UNIQUE_LOGICS, RISK_CONFIG

logging.basicConfig(level=logging.INFO)

def main():
    connector = CoinDCXConnector()
    logic = LogicEvaluator(UNIQUE_LOGICS)
    scanner = InstitutionalScanner(
        connector,
        logic,
        {"min_signal_score": 65}
    )
    risk = RiskManager(RISK_CONFIG)

    while True:
        signals = scanner.scan_all()
        for s in signals:
            print(s)
        time.sleep(300)

if __name__ == "__main__":
    main()