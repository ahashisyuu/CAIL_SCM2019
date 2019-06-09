import json
import random

import pandas as pd

import tokenization
from data_process.split_data import Example, convert_examples


def main():
    data_file = "../data/train.txt"

    all_data = pd.read_csv(data_file, sep="\t")

    num_examples = len(all_data["Quality"].tolist())

    all_examples = []
    for i in range(0, num_examples, 2):
        AB = all_data.iloc[i].tolist()
        AC = all_data.iloc[i+1].tolist()

        assert isinstance(AB[4], str) is True
        assert isinstance(AC[4], str) is True
        example = Example(A=AB[3], B=AB[4], C=AC[4])

        all_examples.append(example)

    random.shuffle(all_examples)
    print("num examples: ", num_examples)
    count = num_examples // 2
    train_examples = all_examples[:int(0.8*count)]
    dev_examples = all_examples[int(0.8*count):]

    # tokenization
    tokenizer = tokenization.FullTokenizer(
        vocab_file="../bert_data/vocab.txt", do_lower_case=True)

    train_ids, train_length = convert_examples(train_examples, tokenizer)
    dev_ids, dev_length = convert_examples(dev_examples, tokenizer)

    print("train max length: {:4}, {:4}, {:4}".format(*train_length))
    print("dev max length: {:4}, {:4}, {:4}".format(*dev_length))

    with open("../data/data_liu.json", "w") as fw:
        json.dump({"train_ids": train_ids, "dev_ids": dev_ids}, fw)


if __name__ == "__main__":
    main()
