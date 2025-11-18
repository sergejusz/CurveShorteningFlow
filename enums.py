from enum import IntEnum


class CallbackArgs(IntEnum):
    """ """
    ROWS = 0
    COLS = 1
    PATH = 2
    MAX_ITERATIONS = 3
    DIAMETER = 4
    SAVE_TO_FILE_COUNTER = 5
    BACKGROUND_COLOR = 6
    FOREGROUND_COLOR = 7
    HISTORY_COLORS = 8
    LAST_CURVE = 9
    GAUSS_BLURRING = 10
    JET_COLORS = 11
    LINE_THICKNESS = 12
    HISTORY_LENGTH = 13

class HistoryViewStyle(IntEnum):
    """ """
    MAX_COUNT = 20
    SKIP_ITERATIONS = 100
