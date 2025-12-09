from dataclasses import dataclass

@dataclass(frozen=True)
class CallbackData:
    rows: int
    cols: int
    path: str
    max_iterations: int
    diameter: float
    save_to_file_counter: int
    background_color: tuple[int, int, int]
    foreground_color: tuple[int, int, int]
    history_colors: list[tuple[int, int, int]]
    first_curve: bool
    last_curve: bool
    gauss_blurring: int
    jet_colors: bool
    line_thickness: int
    history_length: int
    history_skip: int

