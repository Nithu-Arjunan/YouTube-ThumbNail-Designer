import argparse

from graph import graph


def test_graph(user_topic: str) -> None:
    result = graph.invoke({"user_topic": user_topic})

    print("\nThumbnail generation complete")
    print(f"Topic: {result.get('user_topic', user_topic)}")
    print(f"Iterations: {result.get('iterations')}")
    print(f"Rating: {result.get('rating')}")
    print(f"Best rating: {result.get('best_rating')}")
    print(f"Best image: {result.get('best_image_path')}")
    print(f"Final image: {result.get('final_image_path')}")
    print(f"Report: {result.get('report_path')}")

    image_prompt = result.get("image_generation_prompt")
    if image_prompt:
        print("\nImage generation prompt:")
        print(image_prompt)

    reflection = result.get("reflection")
    if reflection:
        print("\nCritic verdict:")
        print(reflection.verdict)


def main():
    parser = argparse.ArgumentParser(description="Test the YouTube thumbnail graph.")
    parser.add_argument(
        "topic",
        nargs="?",
        default="How AI agents will change software jobs",
        help="Video topic to generate a thumbnail for.",
    )
    args = parser.parse_args()
    test_graph(args.topic)


if __name__ == "__main__":
    main()
