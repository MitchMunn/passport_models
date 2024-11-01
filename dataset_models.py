import patito as pt
from typing import Optional


class Image(pt.Model):
    file_name: str
    image_type: str


class TargetItem(pt.Model):
    target_id: int
    bounding_box_top_left_pixel_number: Optional[int] = pt.Field(allow_missing=True)
    bounding_box_bottom_right_pixel_number: Optional[int] = pt.Field(allow_missing=True)


class AIPassportTable(pt.Model):
    image: Image
    target: list[TargetItem]
