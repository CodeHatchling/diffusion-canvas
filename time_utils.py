import time

nesting_depth: int = 0


class Timer:
    def __init__(self, name="Block"):
        self.name = name

    def __enter__(self):
        global nesting_depth
        nesting_depth += 1
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.perf_counter()
        global nesting_depth
        nesting_depth -= 1
        self.interval = self.end - self.start
        print(f"{'. ' * nesting_depth}{self.name} took {self.interval * 1000:.6f} milliseconds")


class TimeBudget:
    def __init__(self, time_limit_seconds):
        self.stop_time = time.perf_counter() + time_limit_seconds
        self.index = 0

    def __iter__(self):
        # Initialize the timer
        self.start_time = time.perf_counter()
        return self

    def __next__(self):
        if time.perf_counter() >= self.stop_time and self.index > 0:
            raise StopIteration
        self.index += 1
        return self.index
