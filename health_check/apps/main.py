from fastapi import FastAPI
from app.utils import check_database
import time

app = FastAPI(title="Health Check API", version="1.0")

@app.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    Returns status of the app, uptime, and DB connectivity.
    """
    start_time = time.time()

    db_status = check_database()

    response = {
        "status": "healthy" if db_status else "degraded",
        "uptime": f"{time.time() - start_time:.2f}s",
        "database": "connected" if db_status else "unreachable",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return response
