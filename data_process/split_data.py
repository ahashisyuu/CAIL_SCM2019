import json
import random
import tokenization


class Example:
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C


def convert_examples(examples, tokenizer):

    a_len = b_len = c_len = 0

    example_tokens = []
    for example_index, example in enumerate(examples):
        a_tokens = tokenizer.tokenize(example.A)
        b_tokens = tokenizer.tokenize(example.B)
        c_tokens = tokenizer.tokenize(example.C)

        if len(a_tokens) > a_len:
            a_len = len(a_tokens)
        if len(b_tokens) > b_len:
            b_len = len(b_tokens)
        if len(c_tokens) > c_len:
            c_len = len(c_tokens)

        a_ids = tokenizer.convert_tokens_to_ids(a_tokens)
        b_ids = tokenizer.convert_tokens_to_ids(b_tokens)
        c_ids = tokenizer.convert_tokens_to_ids(c_tokens)

        example_tokens.append([a_ids, b_ids, c_ids])

    return example_tokens, [a_len, b_len, c_len]


def main():
    data_file = "../data/input.txt"

    all_examples = []

    with open(data_file, encoding='utf-8') as fr:
        count = 0
        for line in fr:
            data_line = json.loads(line)
            A = data_line["A"].split('\n')[2]
            B = data_line["B"].split('\n')[2]
            C = data_line["C"].split('\n')[2]
            count += 1

            example = Example(A=A, B=B, C=C)

            all_examples.append(example)

    print("The number of examples: ", count)

    random.shuffle(all_examples)
    train_examples = all_examples[:int(0.8*count)]
    dev_examples = all_examples[int(0.8*count):]

    # tokenization
    tokenizer = tokenization.FullTokenizer(
        vocab_file="../bert_data/vocab.txt", do_lower_case=True)

    train_ids, train_length = convert_examples(train_examples, tokenizer)
    dev_ids, dev_length = convert_examples(dev_examples, tokenizer)

    print("train max length: {:4}, {:4}, {:4}".format(*train_length))
    print("dev max length: {:4}, {:4}, {:4}".format(*dev_length))

    with open("../data/data.json", "w") as fw:
        json.dump({"train_ids": train_ids, "dev_ids": dev_ids}, fw)


if __name__ == "__main__":
    main()
