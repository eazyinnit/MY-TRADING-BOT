import unittest
import sys
import os

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_bot import bot

class TestTradingBot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.symbol = 'BTC/USDT'
        cls.timeframe = '1h'
        cls.limit = 10
        cls.df = bot.fetch_historical_data(cls.symbol, cls.timeframe, cls.limit)
        cls.preprocessed_df = bot.preprocess_data(cls.df)

    def test_fetch_historical_data(self):
        self.assertFalse(self.df.empty, "Historical data should not be empty")
        self.assertEqual(len(self.df), self.limit, f"Historical data should have {self.limit} rows")

    def test_preprocess_data(self):
        self.assertIn('rsi', self.preprocessed_df.columns, "RSI indicator should be in the dataframe")
        self.assertIn('ema_short', self.preprocessed_df.columns, "Short EMA should be in the dataframe")
        self.assertIn('ema_long', self.preprocessed_df.columns, "Long EMA should be in the dataframe")

    def test_fetch_news_headlines(self):
        headlines = bot.fetch_news_headlines('https://crypto.news/news')
        print(f"Fetched headlines: {headlines}")  # Debugging statement
        self.assertIsInstance(headlines, list, "Headlines should be a list")
        self.assertGreater(len(headlines), 0, "Headlines list should not be empty")

    def test_analyze_sentiment(self):
        sentiment = bot.analyze_sentiment("The market is bullish today.")
        self.assertIsInstance(sentiment, float, "Sentiment should be a float")
        self.assertGreaterEqual(sentiment, -1.0, "Sentiment should be >= -1.0")
        self.assertLessEqual(sentiment, 1.0, "Sentiment should be <= 1.0")

    def test_make_trading_decision(self):
        try:
            bot.make_trading_decision(self.preprocessed_df)
        except Exception as e:
            self.fail(f"make_trading_decision raised an exception: {e}")

    def test_trading_env(self):
        env = bot.TradingEnv(self.preprocessed_df)
        obs = env.reset()
        self.assertEqual(len(obs), len(self.preprocessed_df.columns), "Observation length should match dataframe columns")

    def test_make_trading_decision_buy(self):
        # Mock positive sentiment
        bot.fetch_news_headlines = lambda url: ["The market is bullish today."]
        bot.analyze_sentiment = lambda text: 0.5
        try:
            bot.make_trading_decision(self.preprocessed_df)
            # Check logs or other indicators to verify buy logic was executed
        except Exception as e:
            self.fail(f"make_trading_decision raised an exception during buy logic: {e}")

    def test_make_trading_decision_sell(self):
        # Mock negative sentiment
        bot.fetch_news_headlines = lambda url: ["The market is bearish today."]
        bot.analyze_sentiment = lambda text: -0.5
        try:
            bot.make_trading_decision(self.preprocessed_df)
            # Check logs or other indicators to verify sell logic was executed
        except Exception as e:
            self.fail(f"make_trading_decision raised an exception during sell logic: {e}")

    def test_make_trading_decision_hold(self):
        # Mock neutral sentiment
        bot.fetch_news_headlines = lambda url: ["The market is stable today."]
        bot.analyze_sentiment = lambda text: 0.0
        try:
            bot.make_trading_decision(self.preprocessed_df)
            # Check logs or other indicators to verify hold logic was executed
        except Exception as e:
            self.fail(f"make_trading_decision raised an exception during hold logic: {e}")

if __name__ == '__main__':
    unittest.main()