import nltk
import json
import sys
from nltk.tokenize import word_tokenize


# It transforms a nltk tree into a dictionary
def tree_to_dict(tree):
    tdict = []
    for t in tree:
        if isinstance(t, nltk.Tree) and isinstance(t[0], nltk.Tree):
            tdict.append({ t.label() : tree_to_dict(t)})
        elif isinstance(t, nltk.Tree):
            tdict.append({ t.label() : t[0]})
    return tdict


# get sentences to parse from file
def get_sentences():
    sentences = []
    with open("ParserData\\Sentences.txt", 'r', encoding='utf-8') as file:
        line = file.readline()

        while line:
            sentences.append(line.strip())
            line = file.readline()

    return sentences


# Transforms sentences in tokens
def tokenize_sentences(sentences):

    tokenized_sentences = []
    for index, sentence in enumerate(sentences):
        tokenized_sentences.append(word_tokenize(sentence))

    return tokenized_sentences


if __name__ == "__main__":
    # get sentences as tokens
    sentences = get_sentences()
    tokenized_sentences = tokenize_sentences(sentences)

    # get parser grammar
    italian_grammar = nltk.data.load('file:ParserData\\Italian_SimpleGrammar.cfg')
    parser = nltk.ChartParser(italian_grammar)

    # parse each sentence
    for sentence in tokenized_sentences:
        for tree in parser.parse(sentence):
            # print tree as json data
            dict_tree = tree_to_dict(tree)
            json.dump(dict_tree, sys.stdout)

            # set separator
            json.dump("SENTENCE_SEPARATOR", sys.stdout)


