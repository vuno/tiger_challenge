import time
from functools import wraps
from datetime import datetime, timedelta


class Timer:
    def __init__(
        self,
        time_limit: str,  # HH:MM:SS
        time_format: str = "%H:%M:%S",
        auto_start: bool = False,
    ) -> None:
        t = datetime.strptime(time_limit, time_format)
        self._time_limit = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
        self._start_time = None
        if auto_start:
            self._start()

    def _start(
        self,
    ) -> None:
        self._start_time = datetime.now()

    def exceeded_time_limit(
        self,
    ) -> bool:
        if self._get_elapsed_time() > self._time_limit:
            return True
        else:
            return False

    def _get_elapsed_time(
        self,
    ) -> int:
        curr_time = datetime.now()
        elapsed_time = curr_time - self._start_time

        return elapsed_time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap
