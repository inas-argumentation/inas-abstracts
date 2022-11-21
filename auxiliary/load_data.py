import json
import os

hypo_file = os.path.join(os.path.dirname(__file__), "../data/hypotheses.tsv")
train_set_file = os.path.join(os.path.dirname(__file__), "../data/train_set_indices.txt")
test_set_file = os.path.join(os.path.dirname(__file__), "../data/test_set_indices.txt")
val_set_file = os.path.join(os.path.dirname(__file__), "../data/val_set_indices.txt")
data_set_folder = os.path.join(os.path.dirname(__file__), "../data/abstracts")
DOI_folder = os.path.join(os.path.dirname(__file__), "../data/dataset")

def load_hypotheses(file=hypo_file):
    hypos = {}
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            elements = line.split("\t")
            hypos[int(elements[2])] = {"text": elements[1][:-1], "name": " ".join(elements[0].split(" ")[:-1]), "index": int(elements[2])}
    return hypos

def load_dataset(folder=data_set_folder):
    result = {}
    files = os.listdir(folder)
    for f in files:
        if f == ".gitkeep":
            continue
        with open(folder + f"/{f}", "r") as file:
            title, abstract, sub_labels = file.read().split("\n")
            sub_labels = sub_labels.split(",")
            labels = set([int(x[0]) for x in sub_labels])
            index = int(f[:-4])
            result[index] = {"title": title, "abstract": abstract, "labels": labels, "index": index, "sub_labels": sub_labels}
    return result

def load_DOIs_links_titles_labels(folder=DOI_folder):
    result = {}
    files = os.listdir(folder)
    for f in files:
        if f == ".gitkeep":
            continue
        with open(folder + f"/{f}", "r") as file:
            doi, title, link, sub_labels = file.read().split("\t")
            sub_labels = json.loads(sub_labels)
            labels = set([int(x[0]) for x in sub_labels])
            index = int(f[:-4])
            result[index] = {"title": title, "labels": labels, "sub-labels": sub_labels, "index": index, "doi": doi, "link": link}
    return result

def load_train_val_test_split_indices():
    test_indices = [int(x) for x in open(test_set_file, "r").read().split(",")]
    val_indices = [int(x) for x in open(val_set_file, "r").read().split(",")]
    train_indices = [int(x) for x in open(train_set_file, "r").read().split(",")]
    return train_indices, val_indices, test_indices

def load_annotations():
    with open("data/annotations.json", "r") as f:
        annotations = json.loads(f.read())

    papers = load_dataset()

    for idx1 in annotations:
        title, abstract = papers[int(idx1)]["title"], papers[int(idx1)]["abstract"]
        for idx2 in annotations[idx1]["title annotations"]:
            a = annotations[idx1]["title annotations"][idx2]
            a["text"] = title[a["char_span"][0]:a["char_span"][1]]
        for idx2 in annotations[idx1]["abstract annotations"]:
            a = annotations[idx1]["abstract annotations"][idx2]
            a["text"] = abstract[a["char_span"][0]:a["char_span"][1]]

    return annotations