import color_operations as color_ops
import numpy as np

class ColorInterpolator:
    def __init__(self, fg_color, bg_color, num_levels):
        """
        Creates ColorInterpolator object for given foreground and background colors and
        number of levels that corresponds to number of curves in history buffer.
        :param fg_color: foreground color.
        :param bg_color: background color.
        :param num_levels: number of levels where for level=0 color should be foreground and for
        last level=num_levels-1 should be closer to background color.
        level=0 corresponds to current curve and last level (num_level-1) corresponds
        to oldest curve in history buffer.
        """
        (r_fg, g_fg, b_fg) = fg_color
        (r_bg, g_bg, b_bg) = bg_color
        # linear interpolator setup
        self.arg = [0., num_levels - 1.0]
        # 0 -> foreground color
        # num_levels -1 -> background_color + (foreground_color - background_color)*0.4
        self.r_fcn = [r_fg, r_bg + (r_fg - r_bg) * 0.4]
        self.g_fcn = [g_fg, g_bg + (g_fg - g_bg) * 0.4]
        self.b_fcn = [b_fg, b_bg + (b_fg - b_bg) * 0.4]

    def calculate(self, level):
        """
        Performs linear interpolation to get color for level where level>=0 and level<=num_levels-1.
        :param level: number of curve in history buffer.
        :return color for given level.
        """
        return (
            np.interp(level, self.arg, self.r_fcn),
            np.interp(level, self.arg, self.g_fcn),
            np.interp(level, self.arg, self.b_fcn)
        )
