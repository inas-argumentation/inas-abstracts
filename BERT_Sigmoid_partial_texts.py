import traceback
import warnings
import sys
import time
from os.path import exists
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, logging, get_constant_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from auxiliary.load_data import load_dataset, load_train_val_test_split_indices
from auxiliary.evaluate_predictions import evaluate_predictions
from tqdm import tqdm
from nltk import sent_tokenize

num_labels = 10
run_index = 0
batch_size = 4

# maximum number of training epochs
max_epochs = 25

# determines how many models are trained to get the averaged performance score
num_runs = 10


logging.set_verbosity_error()

#possible_checkpoints = ["bert-base-uncased", "bert-large-uncased", "dmis-lab/biobert-base-cased-v1.1", "dmis-lab/biobert-large-cased-v1.1",
#    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"]
possible_checkpoints = ["dmis-lab/biobert-base-cased-v1.1"]

checkpoint = None
save_name = None

model = tokenizer = data_collator = dataset = dataloader = cross_entropy_weights = None

sentence_selection_modes = ["title", "first", "last"]
sentence_selection_mode = sentence_selection_modes[1]
num_sentences = 2

def set_model_type(checkpoint_index, load_weights=False, parallel=True):
    global checkpoint, save_name, model, tokenizer, data_collator, val_samples_added
    checkpoint = possible_checkpoints[checkpoint_index]
    save_name = f"{checkpoint[checkpoint.rfind('/')+1:]}-{sentence_selection_mode}-{run_index}"

    model = torch.nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels), device_ids=list(range(torch.cuda.device_count())))
    if load_weights:
        model.load_state_dict(torch.load(f"saved_models/sigmoid_partial_texts/{save_name}.pkl"))
    if not parallel:
        model = model.module
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors='pt')

def load_model_and_tokenizer_and_collator_and_datasets(checkpoint_index, parallel=True, load_weights=True):
    set_model_type(checkpoint_index, load_weights, parallel)

    global dataset, dataloader, cross_entropy_weights
    data = load_dataset()
    splits = load_train_val_test_split_indices()
    dataset = INAS_Dataset(data, *splits)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)

    cross_entropy_weights = np.ones(num_labels)
    class_counts = {i: len([1 for x in data.values() if i in x["labels"]]) for i in range(num_labels)}
    num_papers = len(data)
    cross_entropy_weights = np.array([(num_papers / class_counts[i]) ** (1) for i in range(num_labels)])
    cross_entropy_weights = torch.unsqueeze(torch.Tensor(cross_entropy_weights).to("cuda"), dim=0)

def update_loss_avg(new_loss, average):
    if average is None:
        average = new_loss
    else:
        average = 0.9*average + 0.1 * new_loss
    return average

def binary_cross_entropy_with_logits(y_pred, y_true):
    sigmoid_out =  torch.sigmoid(y_pred) * 0.999 + 0.0005
    loss_positive = -torch.sum(torch.log(sigmoid_out) * y_true * cross_entropy_weights, dim=-1)
    loss_negative = -torch.sum(torch.log(1-sigmoid_out) * (1-y_true), dim=-1)

    loss = torch.sum(loss_positive + loss_negative, dim=-1)
    return loss.mean()

def collate_fn(batch):
    texts = [e[0] for e in batch]
    labels = np.array([e[1] for e in batch])
    tokenized_inputs = tokenizer(texts, truncation=True, max_length=512)
    tokenized_inputs["labels"] = labels
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        X = data_collator(tokenized_inputs).to("cuda")
    return X

class INAS_Dataset(Dataset):
    def __init__(self, papers, train_indices, val_indices, test_indices):
        self.papers = papers
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.mode = "train"

        for k in papers.keys():
            if sentence_selection_mode == sentence_selection_modes[0]:
                self.papers[k]["prediction_text"] = f"Title: {self.papers[k]['title']} Abstract: " + "[PAD] " * 20
            else:
                sentences = sent_tokenize(self.papers[k]["abstract"])
                if sentence_selection_mode == sentence_selection_modes[1]:
                    i = 0
                    j = 0
                    s = ""
                    while j < num_sentences:
                        s = s + sentences[i]
                        if len(sentences[i]) > 15:
                            j += 1
                        i += 1
                else:
                    i = len(sentences) - 1
                    j = 0
                    s = ""
                    while j < num_sentences:
                        s = s + sentences[i]
                        if len(sentences[i]) > 15:
                            j += 1
                        i -= 1
                self.papers[k]["prediction_text"] = s

            label = np.zeros(num_labels)
            for l in self.papers[k]["labels"]:
                label[l] = 1
            self.papers[k]["one_hot_label"] = label

    def __len__(self):
        if self.mode == "train":
            return len(self.train_indices)
        elif self.mode == "val":
            return len(self.val_indices)
        else:
            return len(self.test_indices)

    def __getitem__(self, idx):
        if self.mode == "train":
            return self.papers[self.train_indices[idx]]["prediction_text"], self.papers[self.train_indices[idx]]["one_hot_label"]
        elif self.mode == "val":
            return self.papers[self.val_indices[idx]]["prediction_text"], self.papers[self.val_indices[idx]]["one_hot_label"]
        else:
            return self.papers[self.test_indices[idx]]["prediction_text"], self.papers[self.test_indices[idx]]["one_hot_label"]

    def set_mode(self, mode):
        self.mode = mode

def evaluate_model(model, name="Evaluation", print_statistics=True, display_progress=True):
    if display_progress:
        print(f"\nCalculating accuracy on {name.lower()} set:")

    loss_avg = None
    bar = tqdm(desc="Evaluating... Loss: None", total=len(dataloader), position=0, leave=True, file=sys.stdout) if display_progress else None
    model.eval()
    predictions = []
    labels = []
    for batch in dataloader:
        prediction = model(**batch)

        loss = binary_cross_entropy_with_logits(prediction["logits"], batch["labels"])

        loss_avg = update_loss_avg(loss, loss_avg).detach().cpu().numpy()
        if display_progress:
            bar.update(1)
            bar.desc = f"Evaluating... Loss: {loss_avg:<.3f}"

        discrete_predictions = (prediction["logits"].detach().cpu().numpy() > 0.5).astype("int64")

        # In case no label was predicted, treat the label with the maximum predicted score as predicted to be true
        argmaxes = np.argmax(prediction["logits"].detach().cpu().numpy(), axis=-1)
        for i in range(np.shape(discrete_predictions)[0]):
            if np.sum(discrete_predictions[i]) == 0:
                discrete_predictions[i][argmaxes[i]] = 1

        predictions.append(discrete_predictions)
        labels.append(batch["labels"].detach().cpu().numpy())

    if display_progress:
        bar.close()
    macro_F1 = evaluate_predictions(np.concatenate(predictions, axis=0), np.concatenate(labels, axis=0), convert_predictions=False, print_statistics=print_statistics)
    return macro_F1

def train_model(checkpoint_index, load_stuff=True):
    if load_stuff:
        load_model_and_tokenizer_and_collator_and_datasets(checkpoint_index=checkpoint_index, parallel=True, load_weights=False)

    print("Start training...")
    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-6)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0.05)

    dataloader.dataset.set_mode("val")
    max_f1 = evaluate_model(model)
    dataloader.dataset.set_mode("train")
    evals_without_improvement = -3

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for epoch in range(max_epochs):
            print(f"\n\nEpoch {epoch}:")
            model.train()
            loss_avg = None
            bar = tqdm(desc="Loss: None", total=len(dataloader), position=0, leave=True, file=sys.stdout)

            for idx, batch in enumerate(dataloader):
                bar.update(1)
                prediction = model(**batch)

                loss = binary_cross_entropy_with_logits(prediction["logits"], batch["labels"])
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                loss_avg = update_loss_avg(loss, loss_avg).detach().cpu().numpy()
                bar.desc = f"Loss: {loss_avg:<.3f}"

            bar.close()

            dataloader.dataset.set_mode("val")
            f1 = evaluate_model(model)
            dataloader.dataset.set_mode("train")
            evals_without_improvement += 1

            if f1 > max_f1:
                print("save")
                torch.save(model.state_dict(), f"saved_models/sigmoid_partial_texts/{save_name}.pkl")
                max_f1 = f1
                evals_without_improvement = (min(evals_without_improvement, 0))

            if evals_without_improvement == 5:
                break

    t = str(time.time() - start_time)
    t = int(t[:t.index(".")])
    print(f"\nTotal training time: {int(t/60)}:{t%60:02d}")
    print(f"Max F1: {max_f1}")

def train_all_models():
    global run_index
    load_model_and_tokenizer_and_collator_and_datasets(checkpoint_index=0, parallel=True, load_weights=False)
    for checkpoint_index in range(len(possible_checkpoints)):
        current_checkpoint = possible_checkpoints[checkpoint_index]
        print(current_checkpoint)

        for i in range(num_runs):
            run_index = i
            local_save_name = f"{current_checkpoint[current_checkpoint.rfind('/')+1:]}-{sentence_selection_mode}-{run_index}"

            while not exists(f"saved_models/sigmoid_partial_texts/{local_save_name}.pkl"):
                set_model_type(checkpoint_index)
                train_model(checkpoint_index, load_stuff=False)

def test_all_models():
    global run_index
    load_model_and_tokenizer_and_collator_and_datasets(checkpoint_index=0, parallel=True, load_weights=False)
    dataloader.dataset.set_mode("test")
    all_scores = []
    for checkpoint_index in range(len(possible_checkpoints)):
        current_checkpoint = possible_checkpoints[checkpoint_index]
        print(current_checkpoint)
        set_model_type(checkpoint_index, False, True)
        scores = []
        for i in range(num_runs):
            run_index = i
            local_save_name = f"{current_checkpoint[current_checkpoint.rfind('/')+1:]}-{sentence_selection_mode}-{run_index}"
            print(f"\nRun: {run_index}")
            model.load_state_dict(torch.load(f"saved_models/sigmoid_partial_texts/{local_save_name}.pkl"))

            score = evaluate_model(model, print_statistics=True)
            scores.append(score)

        all_scores.append(scores)
    print("\n")
    for i, m in enumerate(possible_checkpoints):
        print(f"{m}:  mean = {np.mean(all_scores[i])} (std = {np.std(all_scores[i])}))  All scores: {all_scores[i]}")

if __name__ == '__main__':
    sentence_selection_mode = sentence_selection_modes[0] # Change this to train the models on the titles/first two sentences/last two sentences
    train_all_models()
    test_all_models()