from datasets import load_dataset
from datasets import IterableDataset

SYSTEM_PROMPT = """you're a helpful assistant."""

XML_COT_FORMAT = """{think}

boxed{{{answer}}}"""


def get_dataset(split="train", sft=False, split_half=None) -> IterableDataset:
    # https://huggingface.co/datasets/openai/gsm8k
    data = load_dataset("openai/gsm8k", "main", cache_dir="./data")
    data = data[split]

    if split_half == "first_half":
        data = data.shard(2, 0)
    elif split_half == "second_half":
        data = data.shard(2, 1)

    if not sft:
        data = data.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": x["question"]},
                ],
                "answer": extract_answer(x["answer"]),  # for reward design
            }
        )
    else:
        data = data.map(
            lambda x: {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": x["question"]},
                    {
                        "role": "assistant",
                        "content": extract_cot(x["answer"]),  # for explicit cot
                    },
                ]
            }
        )
    return data


def extract_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_cot(text: str) -> str:
    if "####" not in text:
        return ""
    cot = text.split("####")
    return XML_COT_FORMAT.format(think=cot[0].strip(), answer=cot[1].strip())


if __name__ == "__main__":
    get_dataset()
