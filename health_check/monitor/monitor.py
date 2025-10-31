import requests
import time
import logging
from datetime import datetime

logging.basicConfig(
    filename="health_monitor.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

HEALTH_URL = "http://127.0.0.1:8000/health"
CHECK_INTERVAL = 10  # seconds

def check_health():
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        if r.status_code == 200 and r.json().get("status") == "healthy":
            logging.info("✅ Service is healthy.")
        else:
            logging.warning(f"⚠️ Service issue: {r.json()}")
    except Exception as e:
        logging.error(f"❌ Failed to connect: {e}")

if __name__ == "__main__":
    print(f"Monitoring {HEALTH_URL} every {CHECK_INTERVAL}s...")
    while True:
        check_health()
        time.sleep(CHECK_INTERVAL)
