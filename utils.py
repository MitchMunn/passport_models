def convert_coords_to_pixel(x, y, image_width):
    pixel = (y - 1) * image_width + x
    return pixel
