
def get_class_color(cls_num):
    """
    Generate a color based on class number for consistent coloring of different classes.
    """
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
             (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)
