import json


def main():
    data_file = "../data/input.txt"
    with open(data_file, encoding='utf-8') as fr:
        count = 0
        for line in fr:
            data_line = json.loads(line)
            print(data_line["A"].split('\n')[2])
            # print('\n\n')
            print(data_line["B"].split('\n')[2])
            print(data_line["C"].split('\n')[2])
            count += 1
            print("\n")

            if count > 5:
                break


if __name__ == "__main__":
    main()
