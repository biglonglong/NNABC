import re
from utils import get_dataset


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    # start with boxed{ and end with }
    pattern = r"boxed\{[^}]*\}"
    matches = [
        re.search(pattern, completion[0]["content"]) for completion in completions
    ]
    return [1.0 if match else 0.0 for match in matches]


def correctness_reward_func(answer, completions, **kwargs) -> list[float]:
    # equal by chars
    responses = [completion[0]["content"] for completion in completions]
    responses_answer = [extract_number_from_boxed_string(r) for r in responses]
    return [
        2.0 if ra == a.replace(" ", "").replace(",", "") else 0.0
        for ra, a in zip(responses_answer, answer)
    ]


def extract_number_from_boxed_string(s):
    # adjust re
    s = s.replace("\\!", "")
    # get number in boxed
    number_match = re.search(r"boxed[^\d]*(\d[\d,]*)", s)
    number = number_match.group(1).replace(",", "") if number_match else None
    return number


REWARD_MAP = {
    "strict_format_reward_func": strict_format_reward_func,
    "correctness_reward_func": correctness_reward_func,
}

if __name__ == "__main__":
    num_check = 2
    data = get_dataset(split="test")  # [18, 3]
    answer = [data[i]["answer"] for i in range(num_check)]
    completions = [
        [{"role": "assistant", "content": "boxed{18}"}] for _ in range(num_check)
    ]

    format_rewards = strict_format_reward_func(completions=completions)
    print("Format Rewards:", format_rewards)
    correctness_rewards = correctness_reward_func(
        answer=answer, completions=completions
    )
    print("Correctness Rewards:", correctness_rewards)
