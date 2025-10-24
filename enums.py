from enum import IntEnum


class CallbackArgs(IntEnum):
    """ """
    ROWS = 0
    COLS = 1
    PATH = 2
    MAXITERATIONS = 3
    SAVETOFILECOUNTER = 4
    BACKGROUNDCOLOR = 5
    FOREGROUNDCOLOR = 6
    HISTORYCOLORS = 7

class HistoryViewStyle(IntEnum):
    """ """
    MAXCOUNT = 20
    SKIPITERATIONS = 100
