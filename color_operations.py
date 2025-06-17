

def convert_to_rgb(color_string):
    if len(color_string) != 6: return (0,0,0)
    return (int(color_string[4:], 16), int(color_string[2:4], 16), int(color_string[:2], 16))

def get_blue_background_color():
    return convert_to_rgb("edf4fd")

def get_blue_palette():
    color_1 = convert_to_rgb("c0d9f8")
    color_2 = convert_to_rgb("94bff4")
    color_3 = convert_to_rgb("68a4ef")
    color_4 = convert_to_rgb("438eeb")
    color_5 = convert_to_rgb("176cd6")
    return [color_1, color_2, color_3, color_4, color_5]

def get_red_background_color():
    return convert_to_rgb("ffebeb")

def get_red_palette():
    color_1 = convert_to_rgb("ffdada")
    color_2 = convert_to_rgb("ffc2c2")
    color_3 = convert_to_rgb("ff9191")
    color_4 = convert_to_rgb("ff5050")
    color_5 = convert_to_rgb("ff0606")
    return [color_1, color_2, color_3, color_4, color_5]


def get_green_background_color():
    return convert_to_rgb("ebffeb")


def get_green_palette():
    color_1 = convert_to_rgb("91ff91")
    color_2 = convert_to_rgb("47ff47")
    color_3 = convert_to_rgb("00e400")
    color_4 = convert_to_rgb("00b400")
    color_5 = convert_to_rgb("007a00")
    return [color_1, color_2, color_3, color_4, color_5]
