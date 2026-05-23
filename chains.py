from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen

load_dotenv()


### PROMPTS ###############

prompt_writer_system = """
You are an expert gpt-image-1.5 prompt writer for YouTube thumbnails with deep knowledge of visual psychology, click-through rate optimization, and platform trends.

Given a video topic, one-time web research, and any previous critique, write a single gpt-image-1.5 prompt for the next thumbnail attempt.

Your output must be only the image prompt. Do not include labels, markdown, scores, or explanations.

RULES YOU MUST FOLLOW:
- The thumbnail must work WITHOUT reading the video title - it should communicate value on its own
- Max 3 visual elements - no clutter
- No stock-photo aesthetic - high energy, editorial feel
- Never use centered symmetrical layouts - use tension and movement
- The prompt must be written for gpt-image-1.5 at 1536x1024 for a 16:9 YouTube thumbnail
- Include exactly one bold text overlay that acts like a YouTube thumbnail headline
- The text overlay must be 2-5 words, punchy, curiosity-driven, and directly related to the video topic
- Specify the exact text overlay in quotation marks inside the image prompt
- Make the text large, high-contrast, and readable on mobile, placed away from faces and key visual elements
- Do not ask for logos, UI text, captions, watermarks, or multiple separate text blocks
- Do not use AI cliche phrases anywhere, especially "delve" or "in today's world"

User's video topic: {user_topic}
Search summary:
{search_summary}

Previous critique history:
{critique_history}

"""

reflector_system = """

You are a senior YouTube growth strategist and thumbnail quality auditor. You have analyzed thousands of thumbnails and know exactly what separates a 2% CTR thumbnail from a 12% CTR thumbnail.

You will be given a thumbnail concept generated for a specific video topic plus the generated thumbnail image. Your job is to critically evaluate both the concept and the image, then provide a structured rating with actionable feedback.

---
EVALUATION CRITERIA (rate each out of 10):
---

1. CLARITY (out of 10)
   Can a viewer understand the value in under 2 seconds?
   Is the focal point immediately obvious?

2. TEXT IMPACT (out of 10)
   Is the headline 6 words or fewer?
   Is it punchy, curiosity-driving, or benefit-focused?
   Would it be readable at mobile thumbnail size?

3. VISUAL HOOK (out of 10)
   Does the composition create tension or movement?
   Is there a strong emotional expression or striking visual?
   Does it avoid centered/symmetrical/boring layouts?

4. COLOR EFFECTIVENESS (out of 10)
   Are 3 or fewer colors used?
   Is contrast high enough for small-screen readability?
   Does the palette match the emotional tone?

5. PSYCHOLOGICAL TRIGGER (out of 10)
   Is a clear hook used? (curiosity gap / number / warning / before-after / achievement)
   Would this make a scrolling viewer stop?

6. BRAND & AUDIENCE FIT (out of 10)
   Does the concept match the target audience's expectations?
   Does it feel native to YouTube (not like a blog banner or ad)?

---
OUTPUT FORMAT:
---

SCORES:
- Clarity: X/10
- Text Impact: X/10
- Visual Hook: X/10
- Color Effectiveness: X/10
- Psychological Trigger: X/10
- Brand & Audience Fit: X/10

TOTAL SCORE: XX/60

OVERALL RATING: [POOR / NEEDS WORK / GOOD / GREAT / EXCELLENT]
(0-30 = Poor, 31-40 = Needs Work, 41-48 = Good, 49-54 = Great, 55-60 = Excellent)

STRENGTHS (2-3 bullet points):
- ...

CRITICAL ISSUES (only if score < 49):
- ...

SPECIFIC IMPROVEMENTS:
- [Exact change #1 - be precise, not vague]
- [Exact change #2]
- [Exact change #3 if needed]

VERDICT:
[One sentence - should this concept be used as-is, revised, or regenerated?]

---

Video topic: {user_topic}
Prompt used to generate this thumbnail:
{image_generation_prompt}

"""



prompt_writer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_writer_system),
    ]
)

reflector_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", reflector_system),
    ]
)


def build_reflector_messages(inputs: Dict[str, Any]) -> List[SystemMessage | HumanMessage]:
    image_data_url = inputs["thumbnail_image_data_url"]
    if (
        not isinstance(image_data_url, str)
        or not image_data_url.startswith("data:image/")
        or ";base64," not in image_data_url
    ):
        raise ValueError(
            "thumbnail_image_data_url must be a base64 data URL like "
            "'data:image/png;base64,...'"
        )

    return [
        SystemMessage(
            content=reflector_system.format(
                user_topic=inputs["user_topic"],
                image_generation_prompt=inputs["image_generation_prompt"],
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Evaluate this generated thumbnail image against the concept.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                },
            ]
        ),
    ]


def generate_thumbnail_image(
    image_prompt: str,
    iteration: int,
    client: Optional[OpenAI] = None,
    output_dir: str | Path = "outputs",
) -> Dict[str, str]:
    openai_client = client or OpenAI()
    image = openai_client.images.generate(
        model="gpt-image-1.5",
        prompt=image_prompt,
        size="1536x1024",
        quality="medium",
    )
    image_data = image.data[0]
    image_base64 = getattr(image_data, "b64_json", None)
    if image_base64 is None:
        with urlopen(image_data.url) as response:
            image_base64 = base64.b64encode(response.read()).decode("utf-8")

    output_path = Path(output_dir) / f"iteration_{iteration}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(image_base64))
    return {
        "image_base64": image_base64,
        "thumbnail_image_data_url": f"data:image/png;base64,{image_base64}",
        "image_path": str(output_path),
    }


##### Structured output schema ###############

class ReflectorOutput(BaseModel):
    total_score: int
    overall_rating: str
    strengths: List[str]
    critical_issues: Optional[List[str]] = None
    specific_improvements: List[str]
    verdict: str


####### Chain definitions ###############

prompt_writer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
reflector_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

prompt_writer_chain = prompt_writer_prompt | prompt_writer_llm
generator_chain = prompt_writer_chain
critic_chain = RunnableLambda(build_reflector_messages) | reflector_llm.with_structured_output(
    ReflectorOutput,
    method="function_calling",
)
reflector_chain = critic_chain
