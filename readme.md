# Hypothesis Identification and Analysis with the INAS Dataset #

This is the code to create the INAS dataset and to reproduce the results from the paper _Linking a Hypothesis Network From the Domain of Invasion Biology to a Corpus of  Scientific Abstracts: The INAS Dataset_.

## 1. Dataset Creation ##

Due to copyright, the abstracts from the INAS dataset can not be published directly. Instead, information about each paper (Title and DOI) is
available in the `data/dataset` folder. The project includes a webscraper to automatically download the corresponding abstracts.
For this, please execute `download_abstracts.py` after installing Firefox and placing the geckodriver executable (https://github.com/mozilla/geckodriver/releases) in the PATH.

## 2. Result Reproduction ##

Different files can be used to reproduce the results from our paper:

* `analyze_dataset.py` can be used to extract dataset statistics and analyze the annotations.
* `NaiveBayes.py` can be used to reproduce the naive Bayes classification results.
* `BERT_Sigmoid.py` and `BERT_Softmax.py` can be used to reproduce the hypothesis classification results for all neural network classifiers
and to create the performance analysis plot with regard to the number of annotations per text.
* `BERT_Sigmoid_partial_texts.py` can be used to analyze the performance of the neural networks on only the first two sentences, the last
two sentences or the title.

The functionality of the programs can currently not be controlled using command line arguments.
It is advised to open the code in a programming environment and access the code to control the program behavior.