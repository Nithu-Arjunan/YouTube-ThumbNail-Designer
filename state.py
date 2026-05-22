from typing import Optional, TypedDict

from chains import ReflectorOutput

class State(TypedDict, total=False):
    user_topic: str
    search_summary: str
    image_generation_prompt: str
    image_base64: str
    thumbnail_image_data_url: str
    image_path: str
    generated_images: list[str]
    reflection: ReflectorOutput
    rating: Optional[int]
    best_rating: Optional[int]
    best_image_path: str
    critique_history: list[str]
    iterations: int
    final_image_path: str
    report_path: str
