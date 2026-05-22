import shutil
from pathlib import Path
from typing import Literal

from tavily import TavilyClient

from chains import critic_chain, generate_thumbnail_image, prompt_writer_chain
from state import State


TARGET_RATING = 55
MAX_ITERATIONS = 5


def web_search_node(state: State) -> State:
    response = TavilyClient().search(state["user_topic"], max_results=5)
    search_results = []
    for hit in response.get("results", []):
        title = hit.get("title", "Untitled")
        summary = hit.get("content") or hit.get("snippet") or hit.get("url", "")
        search_results.append(f"{title}: {summary}")

    return {
        "search_summary": "\n".join(search_results),
        "iterations": 0,
        "rating": 0,
        "critique_history": [],
        "generated_images": [],
    }


def prompt_writer_node(state: State) -> State:
    output = prompt_writer_chain.invoke(
        {
            "user_topic": state["user_topic"],
            "search_summary": state.get("search_summary", ""),
            "critique_history": "\n".join(state.get("critique_history", [])) or "None yet.",
        }
    )
    return {"image_generation_prompt": output.content}


def generator_node(state: State) -> State:
    iteration = state.get("iterations", 0) + 1
    image_result = generate_thumbnail_image(
        state["image_generation_prompt"],
        iteration=iteration,
    )
    return {
        **image_result,
        "iterations": iteration,
        "generated_images": [*state.get("generated_images", []), image_result["image_path"]],
    }


def critic_node(state: State) -> State:
    reflection = critic_chain.invoke(
        {
            "user_topic": state["user_topic"],
            "image_generation_prompt": state["image_generation_prompt"],
            "thumbnail_image_data_url": state["thumbnail_image_data_url"],
        }
    )
    rating = reflection.total_score
    previous_best = state.get("best_rating")
    is_best = previous_best is None or rating > previous_best

    return {
        "reflection": reflection,
        "rating": rating,
        "critique_history": [*state.get("critique_history", []), reflection.verdict],
        "best_rating": rating if is_best else previous_best,
        "best_image_path": state["image_path"] if is_best else state.get("best_image_path"),
    }


def saver_node(state: State) -> State:
    best_image_path = state.get("best_image_path") or state.get("image_path")
    if best_image_path:
        shutil.copyfile(best_image_path, "final.png")

    report = [
        f"# YouTube Thumbnail Report",
        "",
        f"Topic: {state.get('user_topic', '')}",
        f"Best rating: {state.get('best_rating', state.get('rating', 0))}",
        f"Best image: {best_image_path or ''}",
        "",
        "## Final Prompt",
        state.get("image_generation_prompt", ""),
        "",
        "## Critique History",
        *[f"- {item}" for item in state.get("critique_history", [])],
    ]
    Path("report.md").write_text("\n".join(report), encoding="utf-8")
    return {
        "final_image_path": "final.png" if best_image_path else "",
        "report_path": "report.md",
    }


def should_continue(state: State) -> Literal["saver", "prompt_writer"]:
    rating = state.get("rating", 0) or 0
    iterations = state.get("iterations", 0)
    if rating >= TARGET_RATING or iterations >= MAX_ITERATIONS:
        return "saver"
    return "prompt_writer"


generate_thumbnail_concept_node = prompt_writer_node
generate_thumbnail_image_node = generator_node
reflect_thumbnail_node = critic_node
