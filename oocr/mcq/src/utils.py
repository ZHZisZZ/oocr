import random
import re

import datasets


def get_raw_dataset(dataset_name_or_path: str, mode: str = "idx-0", num_proc: int = 8):
    # return: {"question": ..., "answers": []}
    if dataset_name_or_path.startswith("sycophancy_eval_answer"):
        dataset = datasets.load_dataset(
            "json", data_files="oocr/mcq/datasets/sycophancy_eval/answer.jsonl"
        )["train"]
        def sycophancy_map_fn(row):
            return {
                "question": row["prompt"][0]["content"], 
                "answers": [
                    row["base"]["correct_answer"], 
                    row["base"]["incorrect_answer"]
                ],
            }
        dataset = dataset.map(sycophancy_map_fn, num_proc=num_proc, remove_columns=["prompt", "base", "metadata"])
    else:
        raise NotImplementedError

    n_answers = len(dataset[0]["answers"])
    assert n_answers == 2, "only support binary classification now"

    if mode.startswith("idx"):
        idx = int(mode.split("-")[1])
        assert idx in list(range(n_answers))
        target_answer_idxs = [idx] * len(dataset)
    elif mode.startswith("seed"):
        seed = int(mode.split("-")[1])
        rng = random.Random(seed)
        target_answer_idxs = rng.choices(range(n_answers), k=len(dataset))

    dataset = dataset.add_column("target_answer_idx", target_answer_idxs)

    return dataset


def convert_raw_to_mcq(dataset: datasets.Dataset, num_proc: int = 8):
    # mode: ["idx-{idx}", "seed-{seed}"]
    # return: {"mc_question": ..., "mc_answer": ...("A"/"B"), "target_answer_idx": idx, "question": ..., "answers": []}

    dataset_debias = dataset.map(lambda row: {
        "answers": [row["answers"][1], row["answers"][0]],
        "target_answer_idx": 1-row["target_answer_idx"],
        },
        num_proc=num_proc,
    )

    dataset = datasets.concatenate_datasets([dataset, dataset_debias])

    dataset = dataset.map(lambda row: {
        "mc_question": f"{row['question']}\nA: {row['answers'][0]}\nB: {row['answers'][1]}",
        "mc_answer": ["A", "B"][row['target_answer_idx']],
        },
        num_proc=num_proc,
    )

    return dataset


if __name__ == "__main__":
    raw_dataset = get_raw_dataset(dataset_name_or_path="sycophancy_eval_answer")
    dataset = convert_raw_to_mcq(raw_dataset, mode="idx-0")
    breakpoint()
