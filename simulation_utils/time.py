from datetime import datetime
import zoneinfo

MSK = zoneinfo.ZoneInfo("Europe/Moscow")


def msk_now_str():
    return datetime.now(MSK).strftime("%Y-%m-%d-%H%M%SZ")
