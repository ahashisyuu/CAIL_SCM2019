import os
import json

data_dir = "data/"


def txt2tsv(file="", output_file=""):
    with open(data_dir + file, encoding="utf8") as data_fh, \
         open(os.path.join(data_dir, output_file), 'w', encoding="utf-8") as train_fh:
        header = data_fh.readline()
        train_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split('\t')
            train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))


def clean_txt(line=""):
    index = line.find("\n\n")
    if index != - 1:
        line = line[index + 2:]
    line = line.strip("\n").replace("\n", "")

    if '事实如下：' in line:
        index = line.find('事实如下：')
        line = line[index + 5:]
    elif '经审理查明' in line:
        index = line.find('经审理查明')
        line = line[index + 6:]
    elif '案件事实：' in line:
        index = line.find('案件事实：')
        line = line[index + 5:]
    elif '经审理认定' in line:
        index = line.find('经审理认定')
        line = line[index + 6:]
    elif '事实和理由：' in line:
        index = line.find('事实和理由：')
        line = line[index + 6:]
    elif '事实及理由：' in line:
        index = line.find('事实及理由：')
        line = line[index + 6:]
    elif '事实与理由：' in line:
        index = line.find('事实与理由：')
        line = line[index + 6:]
    elif '事实：' in line:
        index = line.find('事实：')
        line = line[index + 3:]
    elif '事实为' in line:
        index = line.find('事实为')
        line = line[index + 4:]
    elif '事实，' in line:
        index = line.find('事实，')
        line = line[index + 3:]
    elif '事实，' in line:
        index = line.find('事实，')
        line = line[index + 3:]
    elif '事实为：' in line:
        index = line.find('事实为：')
        line = line[index + 4:]
    # else:
        # print(line)
        # index = line.find("：")
        # while index != -1:
        #     line = line[index + 1:]
        #     index = line.find("：")
    if len(line) > 255:
        line = line[0: 255]
        # print(line)
        return line
    return line


def clean_txt_v3(line=""):
    index = line.find("\n\n")
    if index != - 1:
        line = line[index + 2:]
    line = line.strip("\n").replace("\n", "")

    if '事实如下：' in line:
        index = line.find('事实如下：')
        line = line[index + 5:]
    elif '经审理查明' in line:
        index = line.find('经审理查明')
        line = line[index + 6:]
    elif '案件事实：' in line:
        index = line.find('案件事实：')
        line = line[index + 5:]
    elif '经审理认定' in line:
        index = line.find('经审理认定')
        line = line[index + 6:]
    elif '事实和理由：' in line:
        index = line.find('事实和理由：')
        line = line[index + 6:]
    elif '事实及理由：' in line:
        index = line.find('事实及理由：')
        line = line[index + 6:]
    elif '事实与理由：' in line:
        index = line.find('事实与理由：')
        line = line[index + 6:]
    elif '事实：' in line:
        index = line.find('事实：')
        line = line[index + 3:]
    elif '事实为' in line:
        index = line.find('事实为')
        line = line[index + 4:]
    elif '事实，' in line:
        index = line.find('事实，')
        line = line[index + 3:]
    elif '事实，' in line:
        index = line.find('事实，')
        line = line[index + 3:]
    elif '事实为：' in line:
        index = line.find('事实为：')
        line = line[index + 4:]
    # else:
        # print(line)
        # index = line.find("：")
        # while index != -1:
        #     line = line[index + 1:]
        #     index = line.find("：")
    if len(line) > 255:
        line = line[-256: -1]
        # print(line)
        return line
    return line


def clean_txt_v2(line=""):
    index = line.find("\n\n")
    if index != - 1:
        line = line[index + 2:]
    line = line.strip("\n").replace("\n", "")
    if '诉讼请求' in line:
        index = line.find('诉讼请求')
        index2 = line.find('。', index)
        line = line[index + 5: index2]
    elif '请求：' in line:
        index = line.find('请求：')
        index2 = line.find('。', index)
        line = line[index + 3: index2]
    elif '请求判令' in line:
        index = line.find('请求判令')
        index2 = line.find('。', index)
        line = line[index + 4: index2]
    elif '要求：' in line:
        index = line.find('要求：')
        index2 = line.find('。', index)
        line = line[index + 3: index2]
    elif '判决：' in line:
        index = line.find('判决：')
        index2 = line.find('。', index)
        line = line[index + 3: index2]
    elif '判令' in line:
        index = line.find('判令')
        index2 = line.find('。', index)
        line = line[index + 3: index2]
    else:
        print(line)
    return line


def json2txt(file="", output_file=""):
    x = 0
    with open(file, 'r', encoding='utf-8') as load_f, \
         open(output_file, 'w', encoding="utf-8") as train_fh:
        # train_fh.write("Quality	#1 ID	#2 ID	#1 String	#2 String\n")
        max_len1 = max_len2 = 0
        for row in load_f:
            load_dict = json.loads(row)
            # A = clean_txt(load_dict['A'])
            # B = clean_txt(load_dict['B'])
            # C = clean_txt(load_dict['C'])
            # train_fh.write("%s\t%s\t%s\t%s\t%s\n" % ('1', x, x + 1, A, B))
            # train_fh.write("%s\t%s\t%s\t%s\t%s\n" % ('0', x, x + 2, A, C))
            A = clean_txt_v3(load_dict['A'])
            B = clean_txt_v3(load_dict['B'])
            C = clean_txt_v3(load_dict['C'])
            load_dict["A"] = A
            load_dict["B"] = B
            load_dict["C"] = C
            json.dump(load_dict, train_fh)
            train_fh.write('\n')
            # train_fh.write("%s\t%s\t%s\t%s\t%s\n" % ('1', x, x + 1, A, B))
            # train_fh.write("%s\t%s\t%s\t%s\t%s\n" % ('0', x, x + 2, A, C))
            if len(load_dict['A']) > max_len1:
                max_len1 = len(load_dict['A'])
            if len(load_dict['B']) > max_len2:
                max_len2 = len(load_dict['B'])
            if len(load_dict['C']) > max_len2:
                max_len2 = len(load_dict['C'])
            x += 3
    print(max_len1)
    print(max_len2)


if __name__ == '__main__':
    json2txt("input.txt", "train.txt")
    txt2tsv("train.txt", "train.tsv")
