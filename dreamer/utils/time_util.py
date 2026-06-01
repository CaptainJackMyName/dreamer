import time


class TimeUtil:
    @staticmethod
    def get_current_time_in_YMDHMS() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())