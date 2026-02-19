import os

# Core
TRADING_DAYS_PER_YEAR = 252

# Financial Modeling Prep
FMP_BASE_URL = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")

# Rate limiting (your plan allows 750/min; keep buffer)
MAX_CALLS_PER_MIN = int(os.getenv("MAX_CALLS_PER_MIN", "740"))
HTTP_TIMEOUT_SEC = int(os.getenv("HTTP_TIMEOUT_SEC", "30"))
HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))

# Concurrency
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "24"))

# Simulation caps / defaults
MAX_TRIALS_CAP = int(os.getenv("MAX_TRIALS_CAP", "1000000"))
DEFAULT_TRIALS = int(os.getenv("DEFAULT_TRIALS", "50000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10000"))

# Plotting
PLOT_SAMPLE_SIZE = int(os.getenv("PLOT_SAMPLE_SIZE", "50000"))
TOP_SHARPE_TO_PLOT = int(os.getenv("TOP_SHARPE_TO_PLOT", "1000"))

# Defaults
DEFAULT_LOOKBACK_DAYS = int(os.getenv("DEFAULT_LOOKBACK_DAYS", "252"))   # 1y
DEFAULT_HOLD_DAYS = int(os.getenv("DEFAULT_HOLD_DAYS", "63"))            # ~3mo
DEFAULT_RF = float(os.getenv("DEFAULT_RF", "0.00"))                      # annualized
DEFAULT_K = int(os.getenv("DEFAULT_K", "10"))
MAX_K = int(os.getenv("MAX_K", "20"))
