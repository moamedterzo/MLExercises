import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import cohen_kappa_score
from utils import pearson_coefficient, spearman_coefficient
import urllib, urllib3
import json

# percorsi dei file annotati o calcolati
input_file = "sem_eval_files/it.test.data.txt"

annotated_similarity_file = "sem_eval_files/annotated_similarity_data.txt"
racanati_similarity_file = "sem_eval_files/racanati_similarity_data.txt"
sansonetti_similarity_file = "sem_eval_files/sansonetti_similarity_data.txt"

annotated_concepts_file = "sem_eval_files/annotated_concepts_data.txt"
racanati_concepts_file = "sem_eval_files/racanati_concepts_data.txt"
sansonetti_concepts_file = "sem_eval_files/sansonetti_concepts_data.txt"

computed_concepts_file = "sem_eval_files/computed_concepts_data.txt"
computed_similarities_file = "sem_eval_files/computed_similarities_data.txt"
synset_mapper_file = "sem_eval_files/SemEval17_IT_senses2synsets.txt"
nasari_vectors_file = "sem_eval_files/mini_NASARI.tsv"


# Mostra le differenze nell'annotazione inerente la similarità tra coppie di parole
def get_similarity_annotations():
    print("Showing agreement coefficients for similarity scores")

    # ottengo le annotazioni
    scores_racanati = get_annotated_similarity(racanati_similarity_file)
    scores_sansonetti = get_annotated_similarity(sansonetti_similarity_file)

    # mostro le differenze
    show_similarity_scores_difference(scores_racanati, scores_sansonetti)


# Mostra le differenze nell'annotazione inerente l'assegnamento dei concetti per coppie di parole
def get_synsets_annotations():
    print("Showing agreement coefficients for concepts scores")

    # ottengo le annotazioni dei contetti
    scores_racanati = get_annotated_concepts(racanati_concepts_file)
    scores_sansonetti = get_annotated_concepts(sansonetti_concepts_file)

    # mostro le differenze
    show_concepts_scores_difference(scores_racanati, scores_sansonetti,show_differences=True)


# si mostra l'indice di correlazione tra due annotazioni
def show_similarity_scores_difference(score1, score2):

    pearson_coefficient(score1, score2)
    spearman_coefficient(score1, score2)

    # stampo quegli score che si differenziano di almeno 2 unità
    for i in range(len(score1)):
        score_1 = score1[i]
        score_2 = score2[i]

        if abs(score_2 - score_1) >= 2:
            print("Different annotated score at row", i + 1)

    print()


# si mostra la vicinanza nell'annotazione dei concetti
def show_concepts_scores_difference(score1, score2, show_cohen=True, show_differences=False):

    if show_cohen:
        # statistica cohen kappa
        stat_annotation_1 = np.reshape(score1, (len(score1) * 2))
        stat_annotation_2 = np.reshape(score2, (len(score2) * 2))

        score = cohen_kappa_score(stat_annotation_1, stat_annotation_2)
        print("Kappa Cohen score:", score)
    else:
        # statistica accuratezza
        total = len(score1) * 2
        accuracy = sum(x[0] == score1[index][0] for index, x in enumerate(score2))
        accuracy += sum(x[1] == score1[index][1] for index, x in enumerate(score2))
        print("Accuracy:", accuracy * 100 / total)

    # mostro quelle righe che presentano una differenza
    if show_differences:
        for i in range(len(score1)):
            score_1 = score1[i]
            score_2 = score2[i]

            if score_1[0] != score_2[0] or score_1[1] != score_2[1]:
                print("Different annotated synset at row", i + 1)

    print()


# avvia la procedura di calcolo delle similarità tra parole
def run_semantic_similarity():
    # ottenimento coppie di parole da analizzare
    pair_words = get_words_to_analize()

    # per ogni coppia di parole si calcola contestualmente lo score di somiglianza
    # e i synset che sono più "vicini" tra loro
    computed_similarities = []
    computed_concepts = []

    for word1, word2 in pair_words:
        # ottengo i vettori Nasari dei concetti per le due parole, se non esistono si salta il calcolo di somiglianza
        vectors1 = get_nasari_vectors(word1)
        vectors2 = get_nasari_vectors(word2)

        if len(vectors1) == 0 or len(vectors2) == 0:
            print("No vector for", word1, "and", word2)
            continue

        # calcolo score e synset
        synset1, synset2, score = get_max_similarity(vectors1, vectors2)

        # normalizzazione delle similarità calcolate in [0,4]
        computed_similarities.append(score * 4)
        computed_concepts.append([synset1, synset2])

    # calcolo dei livelli di agreement tra annotazione umana e quella automatica
    print("Showing agreement for computed scores:")
    annotated_similarity = np.array(get_annotated_similarity(annotated_similarity_file))
    show_similarity_scores_difference(computed_similarities, annotated_similarity)

    # accuratezza per i synset calcolati
    print("\nShowing agreement for computed concepts:")
    annotated_concepts = get_annotated_concepts(annotated_concepts_file)
    show_concepts_scores_difference(computed_concepts, annotated_concepts, False)

    # memorizzazione dei risultati ottenuti
    print("Saving results...")
    save_computed_similarities(pair_words, computed_similarities)
    save_computed_concepts(pair_words, computed_concepts)


# Restituisco i due concetti aventi0 maggiore similarita' tra loro, insieme allo score
def get_max_similarity(vectors1, vectors2):
    max_similarity = 0
    best_concept_1 = None
    best_concept_2 = None

    # Per ogni concetto di entrambe le parole...
    for concept1 in vectors1:
        for concept2 in vectors2:

            # calcolo della cosine similarity
            cos_similarity = cosine_similarity(vectors1[concept1], vectors2[concept2])
            cos_similarity = cos_similarity[0][0]

            # Memorizzo lo score massimo
            if cos_similarity > max_similarity:
                max_similarity = cos_similarity
                best_concept_1 = concept1
                best_concept_2 = concept2

    return best_concept_1, best_concept_2, max_similarity


# restituisce le coppie di parole da analizzare
def get_words_to_analize():

    result = []

    with open(input_file) as src:
        for line in src:
            line = [*line.strip().split("\t")]
            result.append([line[0], line[1]])

    return result


# restituisce le annotazioni di similarità
def get_annotated_similarity(path_file):
    result = []

    with open(path_file) as src:
        for line in src:
            line = [*line.strip().split("\t")]
            result.append(float(line[2]))

    return result


# restituisce le annotazioni dei concetti
def get_annotated_concepts(path_file):
    result = []

    with open(path_file) as src:
        for line in src:
            line = [*line.strip().split("\t")]

            result.append([line[2],line[3]])

    return result


# restituisco il vettore nasari della parola in input
def get_nasari_vectors(word):

    # pulisco la stringa
    word = word.strip()

    word_concepts = []

    # per ottenere tutti i synset della parola bisogna analizzare il file riga per riga
    with open(synset_mapper_file, 'r') as file:
        line = file.readline().strip()

        # indica se si ha trovato la parola
        word_found = False

        while line != '':
            if not word_found:
                # le parole per essere identificate iniziano con il carattere '#'
                if line.startswith("#"):
                    # ottengo la parola e controllo se è uguale a quella che noi stiamo cercando
                    if word == line.replace("#", "").strip():
                        word_found = True
            else:
                # Se la linea inizia con # sono passato ad una nuova parola e pertanto termino la ricerca
                if line.startswith("#"):
                    break
                else:
                    # aggiungo il concetto alla lista
                    word_concepts.append(line.strip())

            # leggo prossima linea
            line = file.readline().strip()

    # recupero i vettori Nasari per i concetti prelevati
    nasari_vectors = {}

    # apro il file che contiene i vettori Nasari
    with open(nasari_vectors_file, encoding="utf8") as file:
        for line in file:
            # ottengo il babel synset ID
            babel_synset_id = line.split("__")[0]

            # Se l'id è lo stesso di uno dei concetti prelevati precedentemente aggiungo il vettore collegato
            if babel_synset_id in word_concepts:

                # ottengo vettore
                vector = line.strip().split("\t")[1:]

                nasari_vectors[babel_synset_id] = np.array(vector).reshape(1, -1)

    return nasari_vectors


# memorizza le similarità calcolate
def save_computed_similarities(sentences, computed_similarities):
    with open(computed_similarities_file, 'w', encoding='utf-8') as file_out:

        for i in range(len(sentences)):
            file_out.write("{}\t{}\t{}\n".format(sentences[i][0], sentences[i][1],computed_similarities[i]))


# memorizza i concetti calcolati
def save_computed_concepts(sentences, computed_concepts):
    with open(computed_concepts_file, 'w', encoding='utf-8') as file_out:

        for i in range(len(sentences)):

            # per ogni concetto prelevo tutte le lessicalizzazioni possibili
            words1 = get_synset_lemmas(computed_concepts[i][0])
            words2 = get_synset_lemmas(computed_concepts[i][1])

            file_out.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(sentences[i][0], sentences[i][1],
                            computed_concepts[i][0], computed_concepts[i][1],words1, words2))


# restituisce le lessicalizzazioni di un babel synset
def get_synset_lemmas(bn_id):

    # chiamata la servizio babelnet
    service_url = 'http://babelnet.io/v5/getSynset'
    params = {
        'id': bn_id,
        'key': "f903ee18-e39e-4514-b7f8-2de5017d9f99",
        'targetLang': 'IT'
    }

    url = service_url + '?' + urllib.parse.urlencode(params)
    http = urllib3.PoolManager()
    response = http.request('GET', url)

    # si analizzano i dati json ricevuti
    data = json.loads(response.data.decode('utf-8'))
    senses = data['senses']

    # si prelevano le lessicalizzazioni del synset
    lemmas = []

    for result in senses:
        lemma = result['properties']['simpleLemma']
        if lemma not in lemmas:
            lemmas.append(lemma)

    return ','.join(lemmas)


if __name__ == "__main__":

    if input("Press 1 for annotation agreement or another key for semantic similarity\n") == "1":
        print("SHOWING AGREEMENTS\n")
        get_similarity_annotations()
        get_synsets_annotations()
    else:
        print("RUNNING ANALYSIS\n")
        run_semantic_similarity()


