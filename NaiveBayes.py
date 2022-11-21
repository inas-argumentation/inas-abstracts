import numpy as np
import scipy.sparse
import pickle
import spacy
import re
from tqdm import tqdm
from scipy.special import softmax
from joblib import load, dump
from sklearn.naive_bayes import MultinomialNB
from auxiliary.evaluate_predictions import evaluate_predictions
from auxiliary.load_data import load_hypotheses
from sklearn.feature_extraction.text import CountVectorizer
from auxiliary.load_data import load_train_val_test_split_indices, load_dataset

index = 0

def create_preprocessed_BOW_dataset():
    def unnecessary_word(word):
        return word.is_stop or word.is_digit or word.is_bracket or word.is_punct or word.is_space or re.match("\d+\.?\d*", word.lemma_) or len(word.lemma_) == 1

    papers = load_dataset()
    train_indices, val_indices, test_indices = load_train_val_test_split_indices()

    sp = spacy.load('en_core_web_sm')

    train_X = []
    train_Y = []
    val_X = []
    val_Y = []
    test_X = []
    test_Y = []

    for index in tqdm(test_indices + train_indices + val_indices):
        paper = papers[index]
        paper_string = paper["title"] + " - " + paper["abstract"]

        s = sp(paper_string)
        s = " ".join([word.lemma_ for word in s if not unnecessary_word(word)])
        y = np.zeros(10)
        for l in paper["labels"]:
            y[l] = 1

        if index in test_indices:
            test_X.append(s)
            test_Y.append(y)
        elif index in train_indices:
            train_X.append(s)
            train_Y.append(y)
        else:
            val_X.append(s)
            val_Y.append(y)

    vectorizer_object = CountVectorizer(stop_words=None, max_df=0.8, min_df=4)

    vectorizer_object.fit(train_X + val_X)
    train_X = vectorizer_object.transform(train_X)
    val_X = vectorizer_object.transform(val_X)
    test_X = vectorizer_object.transform(test_X)

    scipy.sparse.save_npz("data/preprocessed_BOW/train_X.npz", train_X)
    scipy.sparse.save_npz("data/preprocessed_BOW/val_X.npz", val_X)
    scipy.sparse.save_npz("data/preprocessed_BOW/test_X.npz", test_X)
    np.save("data/preprocessed_BOW/train_Y.npy", np.array(train_Y))
    np.save("data/preprocessed_BOW/val_Y.npy", np.array(val_Y))
    np.save("data/preprocessed_BOW/test_Y.npy", np.array(test_Y))
    with open("data/preprocessed_BOW/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer_object, f)

while True:
    try:
        train_X = scipy.sparse.load_npz("data/preprocessed_BOW/train_X.npz")
        train_Y = np.load("data/preprocessed_BOW/train_Y.npy", allow_pickle=True)
        val_X = scipy.sparse.load_npz("data/preprocessed_BOW/val_X.npz")
        val_Y = np.load("data/preprocessed_BOW/val_Y.npy", allow_pickle=True)
        test_X = scipy.sparse.load_npz("data/preprocessed_BOW/test_X.npz")
        test_Y = np.load("data/preprocessed_BOW/test_Y.npy", allow_pickle=True)
        vectorizer = pickle.load(open("data/preprocessed_BOW/vectorizer.pkl", "rb"))
        break
    except:
        create_preprocessed_BOW_dataset()

hypos = load_hypotheses()

# Duplicate samples with multiple correct labels, once with each label.
def convert_multi_label_to_single_label(X, Y):
    new_X = []
    new_Y = []
    for i in range(X.shape[0]):
        for j in range(10):
            if Y[i][j] == 1:
                new_X.append(X[i].toarray())
                new_Y.append(j)
    return scipy.sparse.csr_matrix(np.squeeze(np.array(new_X), axis=1)), np.array(new_Y)

def index_to_word(index):
    x = np.zeros(len(vectorizer.vocabulary_))
    x[index] = 1
    return vectorizer.inverse_transform([x])[0][0]

# Display the tokens that are most likely to occur, given a specific class.
def most_probable_words():
    clf = load_clf()
    for i in range(len(hypos)):
        print(list(hypos.values())[i]["name"])
        print([y[0] for y in sorted([(index_to_word(x), clf.feature_log_prob_[i][x]) for x in np.argpartition(clf.feature_log_prob_[i], -10)[-10:]], key=lambda x: -x[1])], "\n")

# Display the words that are most indicative of a specific class (even if they occur very infrequently).
def most_influential_words():
    clf = load_clf()
    probabilities = softmax(clf.feature_log_prob_, axis=0)
    print()
    for i in range(len(hypos)):
        print(list(hypos.values())[i]["name"])
        print([y[0] for y in sorted([(index_to_word(x), probabilities[i][x]) for x in np.argpartition(probabilities[i], -10)[-10:]], key=lambda x: -x[1])], "\n")

# Display words that are indicative of a specific class while also being fairly frequent.
def most_influential_and_frequent_words(alpha=1):
    clf = load_clf()
    feature_counts = np.sum(clf.feature_count_, axis=0)
    feature_counts = feature_counts/np.max(feature_counts)
    probabilities = softmax(clf.feature_log_prob_, axis=0)
    probabilities[probabilities < 0.3] = 0
    weights = feature_counts * probabilities
    for i in range(len(hypos)):
        print(list(hypos.values())[i]["name"])
        print([y[0] for y in sorted([(index_to_word(x), weights[i][x]) for x in np.argpartition(weights[i], -10)[-10:]], key=lambda x: -x[1])], "\n")
        #print([y[1] for y in sorted([(index_to_word(x), weights[i][x]) for x in np.argpartition(weights[i], -10)[-10:]], key=lambda x: -x[1])], "\n")


def load_clf():
    return load(f"saved_models/NB_{index}.joblib")

def save_clf(clf):
    dump(clf, f"saved_models/NB_{index}.joblib")

def evaluate_clf(clf=None, mode="val", print_statistics=None):
    if clf is None:
        clf = load_clf()
    if mode == "val":
        X, Y = val_X, val_Y
    else:
        X, Y = test_X, test_Y

    y_pred = clf.predict(X)
    if print_statistics is None:
        print_statistics = True if mode=="test" else False
    return evaluate_predictions(y_pred, Y, print_statistics=print_statistics)

def train_clf():
    t_X, t_Y = convert_multi_label_to_single_label(train_X, train_Y)
    max_f1 = 0
    best_setting = [0, False]
    for fit_prior in [True, False]:
        for alpha in np.linspace(0, 5, 101):
            clf = MultinomialNB(fit_prior=fit_prior, alpha=alpha)
            clf.fit(t_X, t_Y)
            f1 = evaluate_clf(clf)

            if f1 > max_f1:
                max_f1 = f1
                best_setting = [fit_prior, alpha]

    print(f"\nTest set performance with fit_prior={best_setting[0]} and alpha={best_setting[1]}:")
    clf = MultinomialNB(fit_prior=best_setting[0], alpha=best_setting[1])
    clf.fit(t_X, t_Y)
    evaluate_clf(clf, "test")
    save_clf(clf)

    print(f"\nTest set performance with fit_prior={best_setting[0]} and alpha={best_setting[1]} and additional training data from validation set:")
    t_X_2, t_Y_2 = convert_multi_label_to_single_label(val_X, val_Y)
    t_X = scipy.sparse.vstack([t_X, t_X_2])
    t_Y = np.concatenate([t_Y, t_Y_2])
    clf.fit(t_X, t_Y)
    evaluate_clf(clf, "test")

if __name__ == '__main__':
    train_clf()
    #evaluate_clf(mode="test")
    #most_probable_words()
    #most_influential_words()
    #most_influential_and_frequent_words()
