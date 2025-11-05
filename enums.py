from enum import IntEnum


class CallbackArgs(IntEnum):
    """ """
    ROWS = 0
    COLS = 1
    PATH = 2
    MAXITERATIONS = 3
    DIAMETER = 4
    SAVETOFILECOUNTER = 5
    BACKGROUNDCOLOR = 6
    FOREGROUNDCOLOR = 7
    HISTORYCOLORS = 8

class HistoryViewStyle(IntEnum):
    """ """
    MAXCOUNT = 20
    SKIPITERATIONS = 100
