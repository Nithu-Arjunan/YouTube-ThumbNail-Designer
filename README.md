# YouTube Thumbnail Designer

Agentic YouTube thumbnail generator built with LangGraph, OpenAI image generation, GPT-4o vision critique, and Tavily web search.

The graph researches a video topic, writes an image prompt, generates thumbnail candidates, critiques each image, loops until the thumbnail is strong enough, then saves the best result.

## Architecture

```text
START
  -> web_search
  -> prompt_writer
  -> generator
  -> critic
  -> saver
  -> END

critic condition:
  -> prompt_writer  if rating is below target
  -> saver          if rating is high enough or max iterations reached
```

Nodes:

- `web_search`: runs one Tavily search and stores `search_summary`.
- `prompt_writer`: writes the image-generation prompt using the topic, search summary, and previous critique.
- `generator`: calls the OpenAI image API, saves an iteration PNG, and increments the loop count.
- `critic`: sends the generated image to GPT-4o vision and stores structured feedback.
- `saver`: copies the best image to `final.png` and writes `report.md`.

## Setup

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```

Install/sync dependencies:

```powershell
uv sync
```

## Run

Use the default topic:

```powershell
uv run python main.py
```

Use your own topic:

```powershell
uv run python main.py "How AI agents will change software jobs"
```

The run makes live Tavily and OpenAI API calls.

## Outputs

Generated files:

- `outputs/iteration_*.png`: each generated attempt
- `final.png`: best selected thumbnail
- `report.md`: final prompt, rating, and critique history

## View The Graph

Print Mermaid:

```powershell
uv run python -c "import graph; print(graph.graph.get_graph().draw_mermaid())"
```

Save graph PNG:

```powershell
uv run python -c "import graph; open('graph.png', 'wb').write(graph.graph.get_graph().draw_mermaid_png())"
```

## Tests

Run the local tests:

```powershell
uv run python -m unittest discover -s tests -v
```

The tests mock image generation and do not spend OpenAI image credits.
