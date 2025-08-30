from __future__ import annotations


import json
import os
import re


from pathlib import Path
from typing import Iterator, Tuple
from ollama import Client
from tqdm import tqdm


SRC_JSONL  = Path("./data/processed/ft/srd_chat_normal.jsonl")
DST_JSONL  = Path("./data/processed/ft/srd_chat_augmented.jsonl")
MODEL_HOST = os.getenv("LLM_BINDING_HOST", "http://localhost:11434") # Ollama server host
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-oss:20b")
QUESTION_PROMPT = (
    "Think of a fitting question based on the given keywords and the given "
    "answer in the context of Dungeons & Dragons. Only return the question.\n"
    "Given keywords: {keywords}\n"
    "Given answer: \"{answer}\""
)
THOUGHT_PROMPT = (""""Act like you are thinking in the context of Dungeons and Dragons like in the example below. You should provide only your thought which you should base on this answer that you want to achieve with your thought: \"{answer}\". An example of a reasoning thought part is as follows: --------------------------BEGIN_EXAMPLE-------------------------- <think> Okay, so I need to find the probability that when I pick A balls out of N, where there are C different colors, the number of each color I pick is exactly a1, a2, ..., aC. Hmm, let's think about how to approach this. First, probability problems often involve combinations. The general formula for probability is the number of favorable outcomes divided by the total number of possible outcomes. So, in this case, the favorable outcomes are the ways to pick exactly a1 of color 1, a2 of color 2, and so on up to aC of color C. The total possible outcomes would be all the ways to pick A balls regardless of color distribution. Let me break it down. The total number of ways to choose A balls from N is the combination of N choose A, which is C(N, A) = N! / (A! (N - A)!). Now, the favorable outcomes. For each color i, we need to choose ai balls from the ni available. Since the choices are independent across colors, the number of ways to do this for all colors would be the product of combinations for each color. So that's C(n1, a1) * C(n2, a2) * ... * C(nC, aC). But wait, is that right? Because when you pick a1 from n1, a2 from n2, etc., and the total number picked is A, which should sum up, right? As long as a1 + a2 + ... + aC = A, then this product gives the number of ways to pick exactly that many of each color. So, the probability should be [C(n1, a1) * C(n2, a2) * ... * C(nC, aC)] divided by C(N, A). That makes sense because each combination of selecting the specific numbers from each color is a favorable case, and the total cases are all possible A-sized groups. Let me check if there are any conditions. Each ai has to be ≤ ni, obviously, because you can't pick more balls of a color than are present. Also, the sum of ai's must equal A, which is given in the problem. If any ai > ni, the combination C(ni, ai) would be zero, so the probability would be zero, which is correct. Alternatively, this is a multivariate hypergeometric distribution problem. The hypergeometric distribution models the probability of k successes in a certain number of draws without replacement from a finite population. The multivariate version extends this to multiple categories, which is exactly the case here with different colors. The formula for the multivariate hypergeometric probability is indeed the product of combinations for each category divided by the combination of the total. Therefore, the probability is the product of combinations C(n_i, a_i) for each color i, divided by the combination C(N, A). So, I think that's the answer. </think> --------------------------END_EXAMPLE-------------------------- """)
EMPTY_PAIR_RE = re.compile(r"^<think>\s*</think>\s*", re.IGNORECASE)
TAG_RE        = re.compile(r"</?think>", re.IGNORECASE)


client = Client(host=MODEL_HOST)


def ask(prompt: str) -> str:
    """Send a user-only prompt, return trimmed response."""
    return client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    ).message.content.strip()


def clean_question(raw: str) -> str:
    """Strip any empty <think> block and residual tags from the question."""
    txt = EMPTY_PAIR_RE.sub("", raw).strip()
    return TAG_RE.sub("", txt).strip()


def clean_thought(raw: str) -> str:
    """Drop the leading empty pair if present and ensure exactly one <think> ... </think> wrapper around the content."""
    txt = EMPTY_PAIR_RE.sub("", raw).strip()
    if not txt.lower().startswith("<think"):
        txt = f"<think> {txt} </think>"
    elif "</think>" not in txt.lower():
        txt = txt.rstrip() + " </think>"
    return txt


def all_records(path: Path) -> Iterator[Tuple[str, str]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            msgs = json.loads(line)["messages"]
            yield msgs[0]["content"], msgs[1]["content"]


def main() -> None:
    total = sum(1 for _ in SRC_JSONL.open(encoding="utf-8"))

    with DST_JSONL.open("w", encoding="utf-8") as out_fh, \
         tqdm(total=total, desc="Augmenting", unit="rec", ncols=80) as bar:

        for keywords, answer in all_records(SRC_JSONL):
            # Generate raw question & thought
            raw_q  = ask(QUESTION_PROMPT.format(keywords=keywords, answer=answer))
            raw_th = ask(THOUGHT_PROMPT.format(answer=answer))

            # Clean them
            question = clean_question(raw_q)
            thought  = clean_thought(raw_th)

            # Compose assistant block with thought and answer
            assistant_text = f"{thought}\n\n{answer}"

            # Write augmented record to a new output file
            out_fh.write(json.dumps({
                "messages": [
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": assistant_text}
                ]
            }, ensure_ascii=False) + "\n")

            bar.update()

    print(f"\nAugmented file written to {DST_JSONL.resolve()}")


if __name__ == "__main__":
    main()
