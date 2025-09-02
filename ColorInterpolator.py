import color_operations as color_ops
import numpy as np

class ColorInterpolator:
    def __init__(self, fg_color, bg_color, num_levels):
        (r_fg, g_fg, b_fg) = fg_color
        (r_bg, g_bg, b_bg) = bg_color
        self.arg = [0., num_levels - 1.0]
        self.r_fcn = [r_fg, r_bg + (r_fg - r_bg) * 0.4]
        self.g_fcn = [g_fg, g_bg + (g_fg - g_bg) * 0.4]
        self.b_fcn = [b_fg, b_bg + (b_fg - b_bg) * 0.4]

    def calculate(self, x):
        return (np.interp(x, self.arg, self.r_fcn), np.interp(x, self.arg, self.g_fcn), np.interp(x, self.arg, self.b_fcn))
