import random

def check_database():
    """
    Simulates a database connection check.
    Returns True if connected, False if not.
    """
    # Simulate random failure for demonstration
    return random.choice([True, True, True, False])
