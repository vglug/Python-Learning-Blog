import logging
import configparser
from datetime import datetime
# Step 1: Read configuration
config = configparser.ConfigParser()
# Configuration file name
config.read('config.ini')
# Get values from config file
log_file = config['LOGGING']['log_file']
log_level = config['LOGGING']['log_level']
# Step 2: Configure logging
logging.basicConfig(
    filename=log_file,
    level=getattr(logging, log_level.upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Step 3: Example program logic
def main():
    logging.info("Program started")

    try:
                # Example calculation
        a = int(input("Enter a number: "))
        b = int(input("Enter another number: "))

        result = a / b
        print(f"Result = {result}")
        logging.info(f"Calculation successful: {a} / {b} = {result}")

    except ZeroDivisionError:
        print("Error: Cannot divide by zero.")
        logging.error("Division by zero error occurred.")
    except Exception as e:
        print("An unexpected error occurred:", e)
        logging.exception("Unexpected error")
    finally:
        logging.info("Program ended")


    
