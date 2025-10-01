from datetime import datetime
import zoneinfo

MSK = zoneinfo.ZoneInfo("Europe/Moscow")

DATETIME_FORMAT = "%Y-%m-%d-%H%M%SZ"


def msk_now_str(format=DATETIME_FORMAT):
    return datetime.now(MSK).strftime(format)


def msk_now():
    return datetime.now(MSK)
