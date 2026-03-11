from datetime import datetime, timedelta
import matplotlib.dates as md

# Function to print out next day in string
def next_day_str(date_str: str) -> str:
    """
    Given a date string in 'YYYY-MM-DD' format, return the next day's date string.
    Example: '2025-10-02' -> '2025-10-03'
    """
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

# Function to print out previous day in string
def previous_day_str(date_str: str) -> str:
    """
    Given a date string in 'YYYY-MM-DD' format, return the previous day's date string.
    Example: '2025-10-02' -> '2025-10-01'
    """
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=-1)).strftime("%Y-%m-%d")

def date_hour_formatter(x, pos):
    dt = md.num2date(x)
    if dt.hour == 0:
        return dt.strftime("%d %b")
    return dt.strftime("%H:%M")