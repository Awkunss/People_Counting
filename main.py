from datetime import datetime, timedelta, timezone
import random
def realtime_log() -> dict:
    vn_tz = timezone(timedelta(hours=7))
    now = datetime.now(vn_tz)
    number = random.randint(1, 10)  # Simulating a random number of people
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "time": time_str,
        "number": number
    }
