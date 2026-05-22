import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from chains import build_reflector_messages, generate_thumbnail_image


class FakeImages:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(b64_json="cG5nLWJ5dGVz")])


class FakeOpenAIClient:
    def __init__(self):
        self.images = FakeImages()


class ThumbnailImageGenerationTests(unittest.TestCase):
    def test_generate_thumbnail_image_uses_gpt_image_and_returns_data_url(self):
        client = FakeOpenAIClient()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_thumbnail_image(
                "dramatic thumbnail prompt",
                iteration=1,
                client=client,
                output_dir=temp_dir,
            )

        self.assertEqual(result["image_base64"], "cG5nLWJ5dGVz")
        self.assertEqual(
            result["thumbnail_image_data_url"],
            "data:image/png;base64,cG5nLWJ5dGVz",
        )
        self.assertEqual(Path(result["image_path"]).name, "iteration_1.png")
        self.assertEqual(
            client.images.calls,
            [
                {
                    "model": "gpt-image-1.5",
                    "prompt": "dramatic thumbnail prompt",
                    "size": "1536x1024",
                    "quality": "medium",
                }
            ],
        )

    def test_generate_thumbnail_image_converts_url_response_to_data_url(self):
        client = FakeOpenAIClient()
        client.images.generate = lambda **kwargs: SimpleNamespace(
            data=[SimpleNamespace(url="https://example.test/image.png")]
        )

        class FakeResponse:
            def read(self):
                return b"png-bytes"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("chains.urlopen", return_value=FakeResponse()):
                result = generate_thumbnail_image(
                    "dramatic thumbnail prompt",
                    iteration=1,
                    client=client,
                    output_dir=temp_dir,
                )

        self.assertEqual(result["image_base64"], "cG5nLWJ5dGVz")
        self.assertEqual(
            result["thumbnail_image_data_url"],
            "data:image/png;base64,cG5nLWJ5dGVz",
        )

    def test_build_reflector_messages_requires_data_url(self):
        with self.assertRaisesRegex(ValueError, "base64 data URL"):
            build_reflector_messages(
                {
                    "user_topic": "AI tools",
                    "generated_thumbnail_concept": "Concept",
                    "thumbnail_image_data_url": "abc123",
                }
            )


if __name__ == "__main__":
    unittest.main()
