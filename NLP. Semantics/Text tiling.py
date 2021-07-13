import statistics
from nltk import word_tokenize
from nltk.corpus import wordnet
import utils

FILE_NAME = "res_ex4/articolo_euro2.txt"
WINDOW_SIZE = 80


# restituisce tutte le parole contenute nell'articolo, sotto forma di token
def get_article_tokens():
    sentences = ''
    with open(FILE_NAME, encoding='utf-8') as file:
        for line in file:
            sentences += ' ' + line.strip()

    initial_words = sentences.lower().split(' ')
    return [token[0] for token in [word_tokenize(word) for word in initial_words] if len(token) > 0]


# suddivide i token in finestre di una determinata grandezza
def get_windows(tokens):

    windows = []

    start_index = 0
    real_window_size = int(len(tokens) / int(len(tokens) / WINDOW_SIZE))
    while True:
        if start_index + real_window_size > len(tokens):
            window_tokens = tokens[start_index:]
            break
        else:
            window_tokens = tokens[start_index:start_index + real_window_size]

        # alla finestra associo i lemmi
        windows.append(utils.get_lemmas_from_tokens_as_counts(window_tokens))
        start_index += real_window_size

    return real_window_size, windows


def compute_window_gaps(windows):

    window_gaps = []
    for i in range(len(windows) - 1):
        window1 = windows[i]
        window2 = windows[i + 1]

        total_score = 0

        # insieme delle parole in comune
        union_keys = set(window1.keys()) & set(window2.keys())
        for word1 in window1.keys():

            # le parole che sono presenti in entrambe le finestre vengono scartate
            if word1 in union_keys:
                continue

            for word2 in window2.keys():
                if word2 in union_keys:
                    continue

                if word1 != word2:
                    # lo score è dato dal peso delle due parole (ossia i conteggi) per la loro dissimilarità
                    sim = get_words_similarity(word1, word2)
                    score = (1 - sim) * window1[word1] * window2[word2]

                    total_score += score

        total_score /= real_window_size ** 2
        window_gaps.append(total_score)

    return window_gaps


# la similarità tra due parole è data dalla coppia dei synset che sono più simili secondo la Wup similarity
def get_words_similarity(word1, word2):
    max_sim = 0

    for syn1 in wordnet.synsets(word1):
        for syn2 in wordnet.synsets(word2):
            sim = wordnet.wup_similarity(syn1, syn2)
            if sim and sim > max_sim:
                max_sim = sim

    return max_sim


if __name__ == "__main__":

    print('Getting tokens...')
    tokens = get_article_tokens()

    # suddivido le parole in gruppi di egual misura
    real_window_size, windows = get_windows(tokens)

    # si calcolano le differenze tra le varie finestre
    print("Computing window gaps for", len(windows), "windows...")
    window_gaps = compute_window_gaps(windows)

    # vengono calcolati gli scostamenti significativi, considerando la media e la deviazione standard
    stddev = statistics.stdev(window_gaps)
    mean = statistics.mean(window_gaps)

    # vengono considerati due tipi di metodi per trovare i boundaries significativi
    boundaries_1 = []
    boundaries_2 = []
    for i in range(len(window_gaps)):
        gap = window_gaps[i]
        if gap > mean + stddev:
            boundaries_1.append(i)
        if gap > mean + stddev/2:
            boundaries_2.append(i)

    print("Stddev:", stddev, "mean:", mean)
    print("Computed gap scores:", window_gaps)
    print("Found boundaries 1:", boundaries_1)
    print("Found boundaries 2:", boundaries_2)

    # scrivo su file le parti di testo trovate
    with open("res_ex4/output.txt", 'w', encoding='utf-8') as file:
        for i in range(len(windows)):
            start = i * real_window_size
            if i == len(windows) - 1:
                window_words = tokens[start:]
            else:
                window_words = tokens[start: (i + 1) * real_window_size]

            window_sentence = ' '.join(window_words)
            file.write(window_sentence)
            file.write("\n")

            if i in boundaries_1:
                file.write("B1 ------------------------------------\n")
            if i in boundaries_2:
                file.write("B2 ------------------------------------\n")







