import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import node
from chains import ReflectorOutput


class ThumbnailNodeTests(unittest.TestCase):
    def test_web_search_node_stores_search_summary(self):
        fake_client = SimpleNamespace(
            search=lambda query, max_results: {
                "results": [
                    {"title": "One", "content": "First result"},
                    {"title": "Two", "snippet": "Second result"},
                ]
            }
        )

        with patch.object(node, "TavilyClient", return_value=fake_client):
            result = node.web_search_node({"user_topic": "AI tools"})

        self.assertEqual(result["iterations"], 0)
        self.assertIn("One: First result", result["search_summary"])
        self.assertIn("Two: Second result", result["search_summary"])

    def test_prompt_writer_node_uses_search_summary_and_critique_history(self):
        fake_chain = SimpleNamespace(
            invoke=lambda inputs: SimpleNamespace(content="DALL-E prompt")
        )

        with patch.object(node, "prompt_writer_chain", fake_chain):
            result = node.prompt_writer_node(
                {
                    "user_topic": "AI tools",
                    "search_summary": "Search context",
                    "critique_history": ["Make text larger"],
                }
            )

        self.assertEqual(result, {"image_generation_prompt": "DALL-E prompt"})

    def test_generator_node_saves_png_and_increments_iteration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                node,
                "generate_thumbnail_image",
                return_value={
                    "image_base64": base64.b64encode(b"png-bytes").decode("ascii"),
                    "thumbnail_image_data_url": "data:image/png;base64,cG5nLWJ5dGVz",
                    "image_path": str(Path(temp_dir) / "iteration_1.png"),
                },
            ) as fake_generate:
                result = node.generator_node(
                    {
                        "image_generation_prompt": "DALL-E prompt",
                        "iterations": 0,
                    }
                )

        fake_generate.assert_called_once_with("DALL-E prompt", iteration=1)
        self.assertEqual(result["iterations"], 1)
        self.assertEqual(result["generated_images"], [result["image_path"]])

    def test_critic_node_appends_history_and_tracks_best_image(self):
        reflection = ReflectorOutput(
            total_score=52,
            overall_rating="GREAT",
            strengths=["Clear focal point"],
            specific_improvements=["Increase contrast"],
            verdict="Use with small edits.",
        )
        fake_chain = SimpleNamespace(invoke=lambda inputs: reflection)

        with patch.object(node, "critic_chain", fake_chain):
            result = node.critic_node(
                {
                    "user_topic": "AI tools",
                    "image_generation_prompt": "DALL-E prompt",
                    "thumbnail_image_data_url": "data:image/png;base64,abc123",
                    "image_path": "outputs/iteration_1.png",
                }
            )

        self.assertEqual(result["rating"], 52)
        self.assertEqual(result["best_rating"], 52)
        self.assertEqual(result["best_image_path"], "outputs/iteration_1.png")
        self.assertEqual(result["critique_history"], ["Use with small edits."])

    def test_should_continue_routes_to_saver_or_prompt_writer(self):
        self.assertEqual(
            node.should_continue({"rating": 54, "iterations": 1}), "prompt_writer"
        )
        self.assertEqual(node.should_continue({"rating": 55, "iterations": 1}), "saver")
        self.assertEqual(
            node.should_continue({"rating": 20, "iterations": 2}), "prompt_writer"
        )
        self.assertEqual(
            node.should_continue({"rating": 20, "iterations": node.MAX_ITERATIONS}),
            "saver",
        )


if __name__ == "__main__":
    unittest.main()
