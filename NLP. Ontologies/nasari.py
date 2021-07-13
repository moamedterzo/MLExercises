import math
import numpy as np
from utils import get_lemmas_from_sentence

nasari_filename = 'Nasari_resources/dd-small-nasari-15.txt'
texts_to_resume_folder_path = 'Nasari_resources/text_documents/'
summary_percentages = [0.1, 0.2, 0.3]


# restituisce due oggetti che servono a supporto dei metodi successivi
def get_small_nasari():

    with open(nasari_filename, encoding='utf-8') as f:

        # dizionario che associa a ciascun concetto un vettore Nasari
        concept_vectors = {}

        # dizionario che associa a ciascuna parola l'elenco dei babel synset nei quali occorre insieme al relativo peso
        word_to_concepts = {}

        for line in f:
            elem_line = line.replace("\n", "").split(';')

            # babel synset ID
            bn_id = elem_line[0]

            # ottenimento parole insieme al relativo peso
            words = [[w.split('_')[0], float(w.split('_')[1])] for w in elem_line[2:] if len(w) > 0 and '_' in w]

            # aggiunta del vettore per il babel synset
            concept_vectors[bn_id] = list(np.array(words)[:,0])

            # aggiunta del babel synset a ciascuna parola presente nel vettore insieme al relativo peso
            for word, weight in words:
                if word not in word_to_concepts:
                    word_to_concepts[word] = []

                word_to_concepts[word].append({'bn_id': bn_id, 'weight': weight})

        return word_to_concepts, concept_vectors


# restituisce i documenti da riassumere
def get_documents():

    documents = []
    for file in [
                 'Life-indoors.txt',
                 'Napoleon-wiki.txt',
                 'Andy-Warhol.txt',
                 'Ebola-virus-disease.txt',
                 ]:
        documents.append(DocumentToResume(texts_to_resume_folder_path + file))

    return documents


# classe utilizzata per rappresentare le informazioni del documento
class DocumentToResume:

    def __init__(self, filename):
        with open(filename, encoding='utf-8') as f:

            self.filename = filename

            # lettura righe
            lines = [line.strip() for line in f.readlines() if line.strip()]
            self.title = lines[0]
            self.paragraphs = lines[1:]

            # ottengo anche i lemmi presenti nel titolo e in ciascun paragrafo
            self.title_bow = get_lemmas_from_sentence(self.title)

            self.paragraphs_bow = []
            for i in range(len(self.paragraphs)):
                self.paragraphs_bow.append(get_lemmas_from_sentence(self.paragraphs[i]))

    # restituisce le righe più importanti in base alla salience
    def get_summary(self, paragraphs_order, percentage):

        # calcolo totale dei caratteri
        tot_characters = len(self.title) + sum(len(par) for par in self.paragraphs)

        # calcolo del numero di caratteri da includere nel riassunto
        characters_to_print = int(tot_characters * (1 - percentage))
        print("Summarizing to", characters_to_print, "characters over", tot_characters)

        # indica gli indici dei paragrafi che hanno superato la fase del riassunto
        index_paragraph_to_include = []

        # indica quanti caratteri sono stati inclusi nel riassunto
        printed_chars = 0

        i = 0
        while True:
            paragraph_index = paragraphs_order.index(i)
            i += 1

            # ottengo numero di caratteri del paragrafo scelto
            paragraph = self.paragraphs[paragraph_index]
            paragraph_chars = len(paragraph)

            # controllo che il nuovo totale di caratteri non superi il numero massimo di caratteri da includere
            printed_chars += paragraph_chars
            if printed_chars < characters_to_print:
                index_paragraph_to_include.append(paragraph_index)
            else:
                break

        # restituisco l'insieme dei paragrafi inclusi nel riassunto, ordinati
        summary = []
        for i in range(len(paragraphs_order)):
            if i in index_paragraph_to_include:
                summary.append(self.paragraphs[i])
            else:
                summary.append("REMOVED -- " + self.paragraphs[i])

        return summary


def resume_document(document, word_to_concepts, concept_vectors, concepts_type='multi'):

    print("\nResuming document:", document.title)

    # ottenimento concetti Nasari del titolo
    title_concepts = get_concepts_from_bow(document.title_bow, word_to_concepts, concepts_type)

    # ottenimento concetti Nasari di ciascun paragrafo
    paragraph_concepts = []
    for paragraph_bow in document.paragraphs_bow:
        par_concepts = get_concepts_from_bow(paragraph_bow,word_to_concepts, concepts_type)
        paragraph_concepts.append(par_concepts)

    # calcolo della salience di ogni paragrafo
    number_of_paragraphs = len(paragraph_concepts)
    paragraphs_salience = np.zeros((number_of_paragraphs, number_of_paragraphs))
    title_salience = np.zeros(number_of_paragraphs)

    for i in range(number_of_paragraphs):
        print("elaborating paragraph", i)

        concepts_i = paragraph_concepts[i]

        # nearness col titolo
        title_salience[i] = get_nearness(concepts_i, title_concepts, concept_vectors)

        # calcolo nearness con gli altri paragragi
        for j in range(number_of_paragraphs):
            if i < j:
                concepts_j = paragraph_concepts[j]
                nearness = get_nearness(concepts_i, concepts_j,concept_vectors)

                paragraphs_salience[i, j] = nearness
                paragraphs_salience[j, i] = nearness

    # post processamento dei dati, si ottiene l'ordine di importanza dei paragrafi
    print()
    paragraphs_order = get_paragraphs_order(paragraphs_salience, title_salience)

    # riassunto del testo in base a differenti percentuali
    print()
    for percentage in summary_percentages:
        summary = document.get_summary(paragraphs_order, percentage)

        with open(document.filename.replace('.txt', '_'+ str(percentage) + "_" + concepts_type +'.txt'), 'w', encoding='utf-8') as out_file:
            for line in summary:
                out_file.write(line)
                out_file.write("\n")


# dalla matrice delle salience e il vettore delle salience rispetto al titolo, si ordinano i paragrafi
def get_paragraphs_order(paragraphs_salience, title_salience):
    paragraphs_order = []

    num_items = np.shape(paragraphs_salience)[0]

    for i in range(num_items):
        # ottengo saliences del paragrafo
        par_saliences = paragraphs_salience[i, :]

        # calcolo media e valore massimo delle salience di una paragrafo verso gli altri paragrafi
        mean = 0
        val_max = 0
        for j in range(num_items):
            if j == i:
                continue
            value = par_saliences[j]
            mean += value
            if value > val_max:
                val_max = value

        mean /= num_items - 1

        # lo score aumenta quanto più è alta la media delle salience, ma risente della salience più alta
        # in questa maniera si preferiscono paragrafi che hanno salience più bilanciate verso gli altri paragrafi
        score = (mean / val_max) * mean

        print("saliences for par {:2,d}: salience score {:15,d}, mean {:15,d}, stddev {:15,d}, maxvalue {}, title score {:15,d}"
              .format(i, int(score), int(mean), int(np.std(par_saliences)), str(math.trunc(val_max)).rjust(14), int(title_salience[i])))

        # aggiunta dello score + salience del titolo
        paragraphs_order.append(score + title_salience[i])

    # ottenimento ordinamento
    return list(np.argsort(-1 * np.array(paragraphs_order)))


# restituisce i concetti associati ad un insieme di lemmi
def get_concepts_from_bow(bow, word_to_concepts, type='multi'):

    result_concepts = []

    if type == "multi":
        # Per questo tipo di calcolo verranno presi tutti i possibili concetti associati a ciascun lemma
        for lemma in bow:
            if lemma in word_to_concepts:
                # per ciascun concetto ove è presente il lemma...
                for concept in word_to_concepts[lemma]:
                    bn_id = concept['bn_id']

                    # si aggiunge il concetto insieme al suo peso fornito dal vettore Nasari associato
                    # per il particolare lemma in questione
                    if bn_id not in result_concepts:
                        result_concepts.append(concept) #aggiunta implicita del relativo peso
                    else:
                        index_word = result_concepts.index(bn_id)
                        result_concepts[index_word]['weight'] += concept['weight']

    else:
        # implementazione naive dove per ciascuna parola si prende il concetto che è più simile agli altri concetti
        # della stessa bag of words (metodo da scartare per il momento)
        words_to_disambiguate = []

        # ottenimento concetti legati alle parole
        for lemma in bow:
            if lemma in word_to_concepts:
                concepts = word_to_concepts[lemma]
                if len(concepts) == 1:
                    result_concepts.append(concepts[0])
                else:
                    words_to_disambiguate.append(concepts)

        # ogni parola che ha più di un concetto deve essere disambiguata, si inizia dalle parole
        # che hanno meno concetti da disambiguare
        for word_concepts in sorted(words_to_disambiguate, key=lambda concepts: len(concepts)):
            best_concept = None
            score_best_concept = -1

            # ottengo il concetto che si avvicina di più a quelli da restituire
            for concept in word_concepts:
                score = get_nearness(result_concepts, [concept], concept_vectors)
                if score > score_best_concept:
                    score_best_concept = score
                    best_concept = concept

            result_concepts.append(best_concept)

    return result_concepts


# restituisce la vicinanza tra due insiemi di concetti
def get_nearness(concepts1, concepts2, concept_vectors):

    # la vicinanza tra i due insiemi è data dalla somma di tutte le vicinanze
    # delle possibili coppie di concetti
    total_nearness = 0

    for concept1 in concepts1:
        for concept2 in concepts2:

            bn_id1 = concept1['bn_id']
            bn_id2 = concept2['bn_id']

            if bn_id1 == bn_id2:
                # se i concetti sono uguali, la vicinanza è massima
                vector_nearness = 1
            else:
                # calcolo la WO dei due concetti
                vector1 = concept_vectors[bn_id1]
                vector2 = concept_vectors[bn_id2]

                vector_nearness = get_weighted_overlap(vector1, vector2)

            # la vicinanza viene pesata in base alle parole associate ai due concetti
            weight = concept1['weight'] * concept2['weight']
            total_nearness += vector_nearness * weight

    return total_nearness


# implementa la weighted overlap di due concetti rappresentati tramite vettori Nasari
def get_weighted_overlap(concept1, concept2):

    # intersezioni delle feature in comune
    intersection = set(concept1).intersection(concept2)
    if len(intersection) == 0:
        return 0

    # vedasi la formula del weighted overlap
    numeratore = 0
    for word in intersection:
        numeratore += math.sqrt(concept1.index(word) + concept2.index(word) + 2)

    denominatore = 0
    for i in range(len(intersection)):
        denominatore += math.sqrt(2 * (i + 1))

    return numeratore / denominatore


if __name__ == '__main__':

    # inizializza le variabili di supporto per l'ottenimento dei vettori Nasari
    word_to_concepts, concept_vectors = get_small_nasari()

    # ottenimento dei documenti
    documents = get_documents()

    # riassunto di ciascun documento
    for document in documents:
        resume_document(document,word_to_concepts, concept_vectors)







