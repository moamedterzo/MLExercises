import csv
import math
from nltk.corpus import wordnet as wn
from utils import pearson_coefficient, spearman_coefficient

words_file_path = "res_conceptual_similarity/WordSim353.csv"


# Restuisce la lunghezza massima di un synset dal nodo radice
def custom_max_depth(s1):

    hypernyms = s1.hypernyms()

    if not hypernyms or len(hypernyms) == 0:
        return 1 # Diretto discendente del nodo radice
    else:
        return 1 + max(custom_max_depth(h) for h in hypernyms)


# Questo valore rappresenta la lunghezza massima dell'albero dei sensi (il valore è memorizzato in quanto è statico)
#wn_maxdepth = max(custom_max_depth(ss) for ss in wn.all_synsets())
wn_maxdepth = 20


# Restituisce il lowest common subsumer e le distanze dai due rispettivi synset
def compute_lcs(s1, s2):

    # se i due synset sono uguali allora la distanza è zero
    if s1 == s2:
        return s1, 0, 0

    # queste due liste contengono i synset iperonimi dei due synset dati in input
    # l'algoritmo terminerà quando un elemento della prima lista comparirà anche nella seconda lista
    # o quando si sarà raggiunta la lunghezza massima possibile
    list_s1 = []
    list_s2 = []

    # ogni elemento della lista contiene gli iperonimi dei synset presenti nell'elemento della lista precedente
    list_s1.append((s1,))
    list_s2.append((s2,))

    counter = 0

    # la distanza massima non può superare il doppio della profondità massima dell'albero
    while counter < 2 * wn_maxdepth:
        # calcolo i nodi iperonimi dello strato precedente
        hypernyms_s1 = set()
        for el in list_s1[-1]:
            hypernyms_s1 |= set(el.hypernyms())
        list_s1.append(hypernyms_s1)

        # controllo se gli iperonimi dell'ultimo strato sono presenti nella seconda lista
        for i in range(len(list_s2)):
            level_s2 = list_s2[i]

            for frontier_element in list_s1[-1]:
                if frontier_element in level_s2:
                    return frontier_element, len(list_s1) - 1, i

        # se non ho trovato nulla, espando lo strato della seconda lista
        # ed effettuo gli stessi controlli in maniera inversa
        hypernyms_s2 = set()
        for el in list_s2[-1]:
            hypernyms_s2 |= set(el.hypernyms())
        list_s2.append(hypernyms_s2)

        for i in range(len(list_s1)):
            level_s1 = list_s1[i]

            for frontier_element in list_s2[-1]:
                if frontier_element in level_s1:
                    return frontier_element, i, len(list_s2) - 1

        counter += 1

    return None


# Calcola il percorso più breve per due synset
def custom_shortest_path_distance(s1, s2):

    # il percorso più breve viene calcolato ottenento il LCS
    # se non esiste, si sommano le distanze dei due nodi dal nodo radice
    result = compute_lcs(s1, s2)

    if result:
        return result[1] + result[2]
    else:
        return custom_max_depth(s1) + custom_max_depth(s2)


# Calcola la similarità di due sensi, dato il tipo di similarità da calcolare
def compute_sense_similarity(s1, s2, sim_measure):
    if sim_measure == "WP":

        # Wu & Palmer similarity measure
        result = compute_lcs(s1, s2)

        # se i due synset non hanno un LCS, allora hanno similarità pari a zero
        if result is None:
            return 0

        # ottengo distanze dal lcs e distanza dalla radice del lcs
        lcs, d1, d2 = result
        lcs_max_depth = custom_max_depth(lcs)

        # calcolo la similarità in base alla formula della misura
        wc_similarity = 2 * lcs_max_depth / (d1 + d2 + 2 * lcs_max_depth)

        return wc_similarity

    elif sim_measure == "SP":

        # Shortest path similarity measure
        return (2 * wn_maxdepth - custom_shortest_path_distance(s1, s2)) / (2 * wn_maxdepth)

    elif sim_measure == "LC":

        # Leakcock & Chodorow similarity measure
        return (-math.log((custom_shortest_path_distance(s1, s2) + 1) / (2 * wn_maxdepth + 1))) / math.log(2 * wn_maxdepth + 1)


# calcola la similarità tra due parole, definita come la similarità massima possibile
# rispetto alle combinazioni di sensi che possono assumere le due parole
def compute_word_similarity(w1, w2, sim_measure):

    max_similarity = 0
    for s1 in wn.synsets(w1):
        for s2 in wn.synsets(w2):
            sim = compute_sense_similarity(s1, s2, sim_measure)
            if sim > max_similarity:
                max_similarity = sim

    return max_similarity


if __name__ == "__main__":

    # Viene calcolata la similarità tra coppie di parole in base a differenti indici,
    # i risultati saranno comparati con le annotazioni presenti in un file
    similarity_wordsim = []
    similarity_WP = []
    similarity_SP = []
    similarity_LC = []

    print("Computing words similarity...")

    # lettura del file csv
    with open(words_file_path, newline='') as csvfile:
        wordsim_reader = csv.reader(csvfile, delimiter=',')

        first = True
        for row in wordsim_reader:
            # l'intestazione viene saltata
            if first:
                first = False
                continue

            # i valori annotati sono normalizzati in [0,1]
            similarity_wordsim.append(float(row[2]) / 10)

            # calcolo degli indici di similarità
            similarity_WP.append(compute_word_similarity(row[0], row[1], 'WP'))
            similarity_SP.append(compute_word_similarity(row[0], row[1], 'SP'))
            similarity_LC.append(compute_word_similarity(row[0], row[1], 'LC'))

    # Vengono mostrati i risultati di correlazione tra i risultati calcolati e quelli annotati
    print()

    print("-- WU & PALMER")
    pearson_coefficient(similarity_wordsim, similarity_WP)
    spearman_coefficient(similarity_wordsim, similarity_WP)
    print()

    print("-- SHORTEST PATH")
    pearson_coefficient(similarity_wordsim, similarity_SP)
    spearman_coefficient(similarity_wordsim, similarity_SP)
    print()

    print("-- LEAKCOCK & CHODOROW")
    pearson_coefficient(similarity_wordsim, similarity_LC)
    spearman_coefficient(similarity_wordsim, similarity_LC)
    print()
