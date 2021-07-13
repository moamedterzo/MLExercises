from nltk.corpus.reader import Lemma
from nltk.corpus import semcor, wordnet
from sklearn.metrics import accuracy_score
import re
from utils import get_lemmas_from_sentence, get_synset_context

sentences_file_path = "res_conceptual_similarity/sentences.txt"
sem_cor_number_sentences = 50


# disambigua i termini polisemici presenti nel file sentences.txt
def disambiguate_sentences():

    # ottengo da file le frasi da trattare
    sentences = []
    with open(sentences_file_path) as file_sentences:
        line = file_sentences.readline().strip()
        while line != '':
            sentences.append(line)
            line = file_sentences.readline().strip()

    # analizzo tutte le frasi ottenute
    for i in range(0, len(sentences)):

        sentence = sentences[i]

        # provvedo a mostrare in output il risultato della disambiguazione
        word = get_word_to_disambiguate(sentence)

        print("Sentence: " + sentence.replace("*" + word + "*", word))
        print("Word to disambiguate: " + word)

        # ottengo il miglior synset per la parola da disambiguare nella frase
        best_synset = simplified_lesk(sentence)

        if best_synset:
            print("Best sense found: ", best_synset, " -> " + best_synset.definition())

            # ottengo il sinonimo e lo rimpiazzo nella frase
            synonim = get_word_synonim(word, best_synset)

            if synonim:
                sentence = sentence.replace("*" + word + "*", synonim.upper())
                print("Synonym = " + synonim + " -> ", sentence)
            else:
                print("The word has no synonim")
        else:
            print("No synset found")

        print()


# disambigua i termini polisemici ottenuti dal corpus semcor
def semcore_disambiguation():

    # ottengo frasi e parole da disambiguare
    sentences_list, extracted_nouns = semcor_sentences_extraction()

    target_senses, lesk_obtained_best_senses = [], []
    for i in range(0, len(sentences_list)):

        print()
        print(sentences_list[i])

        for noun in extracted_nouns[i]:
            # prelevo la parola di disambiguare
            word_to_disambiguate = noun[0][0]

            # uso l'algoritmo Lesk
            predicted_sense = str(simplified_lesk(sentences_list[i], word_to_disambiguate))
            lesk_obtained_best_senses.append(predicted_sense)

            # mostro i risultati della disambiguazione per la frase in questione
            true_sense = str(noun.label().synset())
            target_senses.append(true_sense)

            print("- Word to disambiguate: " + word_to_disambiguate)
            print("Target sense =", true_sense, ", predicted =",predicted_sense,"-> OK" if true_sense == predicted_sense else "WRONG")


    # mostro l'accuratezza dell'algoritmo
    accuracy = accuracy_score(lesk_obtained_best_senses, target_senses)

    print("**************\n")
    print("Accuracy score for the SemCor sentences=" + str(accuracy))


# estrae frasi e parole da disambiguare dal corpus semcor
def semcor_sentences_extraction():

    sentences_list = []
    extracted_nouns = []

    print("Extracting Semcor sentences...\n")
    semcor_sentences = semcor.sents()

    # estraggo le prime tot frasi dal corpus
    for i in range(0, sem_cor_number_sentences):

        elem = list(filter(lambda sentence_tree:
                           isinstance(sentence_tree.label(), Lemma) and sentence_tree[0].label() == "NN",
                           semcor.tagged_sents(tag='both')[i]))
        if elem:
            # estraggo il primo e l'ultimo sostantivo
            extracted_nouns.append([elem[0], elem[-1]])

            # estraggo la frase
            sentences_list.append(" ".join(semcor_sentences[i]))

    return sentences_list, extracted_nouns


# implementazione dell'algoritmo Lesk
def simplified_lesk(sentence, word_to_disambiguate=None):

    # ottengo l'insieme dei lemmi per la frase, più la parola da disambiguare e il suo PoS tag
    context, word_to_disambiguate, word_pos_tag = get_lemmas_from_sentence(sentence,
                                                                           get_only_lemmas=False,
                                                                           word_to_disambiguate=word_to_disambiguate)

    # ottengo i synset candidati per la parola da disambiguare
    candidate_synsets = wordnet.synsets(word_to_disambiguate, pos=word_pos_tag)

    # se non ci sono synset, si prova a non considerare il pos tag della parola
    if not candidate_synsets:
        candidate_synsets = wordnet.synsets(word_to_disambiguate)

    if not candidate_synsets:
        return None

    # si restituisce il synset che possiede maggiori termini in comune col contesto della frase
    best_sense = None
    max_overlap = -1

    for sense in candidate_synsets:
        # ottengo il contesto del synset (vedi definizione metodo)
        signature = get_synset_context(sense, True)

        # si calcola lo score
        overlap = len(signature.intersection(context))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense

    return best_sense


# individua la parola che deve essere disambiguata
def get_word_to_disambiguate(sentence):
    for word in sentence.split(" "):
        if word.startswith("*"):
            return re.sub(r'[^\w\s]', '', word)


# restituisce un lemma sinonimo (secondo il synset) della parola in input
def get_word_synonim(word, synset):

    for l in synset.lemma_names():
        if l.lower() != word.lower():
            return l.replace("_", " ")

    return None


if __name__ == '__main__':

    print("PARTE 1")
    print("________________________________________________________\n")
    disambiguate_sentences()

    print("________________________________________________________\n ")
    print("PARTE 2")
    semcore_disambiguation()


