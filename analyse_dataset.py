import numpy as np
from auxiliary import load_data
from transformers import AutoTokenizer
from nltk import sent_tokenize
from sklearn.neighbors import KernelDensity
from auxiliary.load_data import load_annotations
import matplotlib.pyplot as plt

tag_names = ['hypothesis statement', 'hypothesis fragment', 'implicit hypothesis statement', 'hypothesis name']

def predict_KR(train_data, input, kernel_width, leave_out=[]):
    train_x = np.delete(train_data[:, 0], leave_out)
    train_y = np.delete(train_data[:, 1], leave_out)
    weights = np.exp(-np.square(train_x - input) / kernel_width)
    prediction = np.sum(train_y * weights)/np.sum(weights)
    return prediction

def analyze_annotations():
    annotations = list(load_annotations().values())

    p_title = len([x for x in annotations if len(x["title annotations"]) > 0])/len(annotations)
    print(f"\nPercentage of titles that have annotated spans in them: {p_title*100}%")
    print("Broken down into individual types:")
    print(f"Hypothesis statement: {len([x for x in annotations if len([y for y in x['title annotations'].values() if y['type'] == tag_names[0]]) > 0])/len(annotations)}")
    print(f"Hypothesis fragment: {len([x for x in annotations if len([y for y in x['title annotations'].values() if y['type'] == tag_names[1]]) > 0])/len(annotations)}")
    print(f"Implicit hypothesis: {len([x for x in annotations if len([y for y in x['title annotations'].values() if y['type'] == tag_names[2]]) > 0])/len(annotations)}")
    print(f"Hypothesis name: {len([x for x in annotations if len([y for y in x['title annotations'].values() if y['type'] == tag_names[3]]) > 0])/len(annotations)}")

    p_abstract = len([x for x in annotations if len(x["abstract annotations"]) > 0])/len(annotations)
    print(f"\nPercentage of abstracts that have annotated spans in them: {p_abstract*100}%")
    print("Broken down into individual types:")
    print(f"Hypothesis statement: {len([x for x in annotations if len([y for y in x['abstract annotations'].values() if y['type'] == tag_names[0]]) > 0])/len(annotations)}")
    print(f"Hypothesis fragment: {len([x for x in annotations if len([y for y in x['abstract annotations'].values() if y['type'] == tag_names[1]]) > 0])/len(annotations)}")
    print(f"Implicit hypothesis: {len([x for x in annotations if len([y for y in x['abstract annotations'].values() if y['type'] == tag_names[2]]) > 0])/len(annotations)}")
    print(f"Hypothesis name: {len([x for x in annotations if len([y for y in x['abstract annotations'].values() if y['type'] == tag_names[3]]) > 0])/len(annotations)}")

    p_samples = len([x for x in annotations if len(x['abstract annotations']) + len(x['title annotations']) > 0]) / len(annotations)
    print(f"\nPercentage of titles+abstracts that have annotated spans in them: {p_samples * 100}%")
    print("Broken down into individual types:")
    print(f"Hypothesis statement: {len([x for x in annotations if len([y for y in list(x['abstract annotations'].values()) + list(x['title annotations'].values()) if y['type'] == tag_names[0]]) > 0]) / len(annotations)}")
    print(f"Hypothesis fragment: {len([x for x in annotations if len([y for y in list(x['abstract annotations'].values()) + list(x['title annotations'].values()) if y['type'] == tag_names[1]]) > 0]) / len(annotations)}")
    print(f"Implicit hypothesis: {len([x for x in annotations if len([y for y in list(x['abstract annotations'].values()) + list(x['title annotations'].values()) if y['type'] == tag_names[2]]) > 0]) / len(annotations)}")
    print(f"Hypothesis name: {len([x for x in annotations if len([y for y in list(x['abstract annotations'].values()) + list(x['title annotations'].values()) if y['type'] == tag_names[3]]) > 0]) / len(annotations)}")

    n_title = sum([len(x['title annotations']) for x in annotations]) / len(annotations)
    print(f"\nAverage number of annotations per title: {n_title}")
    print("Broken down into individual types:")
    print(f"Hypothesis statement: {sum([len([y for y in x['title annotations'].values() if y['type'] == tag_names[0]]) for x in annotations]) / len(annotations)}")
    print(f"Hypothesis fragment: {sum([len([y for y in x['title annotations'].values() if y['type'] == tag_names[1]]) for x in annotations]) / len(annotations)}")
    print(f"Implicit hypothesis: {sum([len([y for y in x['title annotations'].values() if y['type'] == tag_names[2]]) for x in annotations]) / len(annotations)}")
    print(f"Hypothesis name: {sum([len([y for y in x['title annotations'].values() if y['type'] == tag_names[3]]) for x in annotations]) / len(annotations)}")

    n_abstract = sum([len(x['abstract annotations']) for x in annotations]) / len(annotations)
    print(f"\nAverage number of annotations per abstract: {n_abstract}")
    print("Broken down into individual types:")
    print(f"Hypothesis statement: {sum([len([y for y in x['abstract annotations'].values() if y['type'] == tag_names[0]]) for x in annotations]) / len(annotations)}")
    print(f"Hypothesis fragment: {sum([len([y for y in x['abstract annotations'].values() if y['type'] == tag_names[1]]) for x in annotations]) / len(annotations)}")
    print(f"Implicit hypothesis: {sum([len([y for y in x['abstract annotations'].values() if y['type'] == tag_names[2]]) for x in annotations]) / len(annotations)}")
    print(f"Hypothesis name: {sum([len([y for y in x['abstract annotations'].values() if y['type'] == tag_names[3]]) for x in annotations]) / len(annotations)}")

    n_samples = sum([len(x['title annotations']) + len(x['abstract annotations']) for x in annotations]) / len(annotations)
    print(f"\nAverage number of annotations per complete sample: {n_samples}")
    print("Broken down into individual types:")
    print(f"Hypothesis statement: {sum([len([y for y in list(x['title annotations'].values()) + list(x['abstract annotations'].values()) if y['type'] == tag_names[0]]) for x in annotations]) / len(annotations)}")
    print(f"Hypothesis fragment: {sum([len([y for y in list(x['title annotations'].values()) + list(x['abstract annotations'].values()) if y['type'] == tag_names[1]]) for x in annotations]) / len(annotations)}")
    print(f"Implicit hypothesis: {sum([len([y for y in list(x['title annotations'].values()) + list(x['abstract annotations'].values()) if y['type'] == tag_names[2]]) for x in annotations]) / len(annotations)}")
    print(f"Hypothesis name: {sum([len([y for y in list(x['title annotations'].values()) + list(x['abstract annotations'].values()) if y['type'] == tag_names[3]]) for x in annotations]) / len(annotations)}")

    print(f"\nPercentage of papers that provide a hypothesis statement or name: {len([x for x in annotations if len([y for y in list(x['title annotations'].values()) + list(x['abstract annotations'].values()) if y['type'] == tag_names[0] or y['type'] == tag_names[3]]) > 0]) / len(annotations)}")

    positions = [[], [], [], []]
    for a in annotations:
        for i in range(4):
            positions[i] += [np.mean(x["word_span"])/a['num_words_abstract'] for x in a['abstract annotations'].values() if x["type"] == tag_names[i]]
    positions = [np.array(p).reshape(-1, 1) for p in positions]

    density_models = [KernelDensity(kernel="gaussian", bandwidth=0.1).fit(p) for p in positions]
    input = np.linspace(0, 1, 101).reshape(-1, 1)
    results = [np.exp(x.score_samples(input)) for x in density_models]

    for i, r in enumerate(results):
        plt.plot(input, r, label=tag_names[i])
    plt.legend()
    plt.show()

def analyse_dataset():
    dataset = load_data.load_dataset()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    num_sentences = []
    num_tokens = []
    num_classes = []
    for d in dataset.values():
        sentences = [s for s in sent_tokenize(d["abstract"]) if len(s) > 15]
        num_sentences.append(len(sentences))
        tokens = tokenizer.tokenize(d["title"] + " " + d["abstract"])
        num_tokens.append(len(tokens))
        num_classes.append(len(d["labels"]))

    print(f"Average number of sentences: {np.mean(num_sentences)}")
    print(f"Average number of tokens: {np.mean(num_tokens)}")
    print(f"Percentage of texts with more than 510 tokens: {len([x for x in num_tokens if x > 510]) / len(num_tokens)}")
    print(f"Percentage of texts with more than 1 class: {len([x for x in num_classes if x > 1]) / len(num_classes)}")

def export_files_for_annotation():
    annotations = load_annotations()
    data = load_data.load_dataset()
    for i in annotations:
        string = "-".join([f"{int(x):02d}" for x in data[int(i)]["labels"]]) + f"-paper_{int(i):03d}.txt"
        print(string)
        with open(f"exports/{string}", "w+") as f:
            f.write(data[int(i)]["title"] + "\n" + data[int(i)]["abstract"])

if __name__ == '__main__':
    analyse_dataset()
    analyze_annotations()
