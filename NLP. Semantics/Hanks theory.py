from collections import defaultdict
import nltk
import utils
from nltk.corpus import treebank
nltk.download('brown')

# indica il verbo oggetto dell'analisi
CHOSEN_WORD = "say"

# numero massimo di frasi da prelevare dal corpus
MAX_SENTS = 10000

# distanza del supersenso dalla radice dei sensi in Wordnet
SENSE_DEPTH = 2

if __name__ == "__main__":

    # array che conterrà i sensi degli argomenti per il verbo in questione
    verb_arguments_senses = []

    # prelevo le frasi dal corpus...

    sentences = treebank.parsed_sents()
    print('Ottenendo le frasi...')
    for sentence in sentences[:MAX_SENTS]:

        # ottengo le PoS della frase
        tagged_sentence = sentence.pos()

        # per ciascuna PoS...
        for i in range(len(tagged_sentence)):
            current_sent_part = tagged_sentence[i]
            pos_tag = current_sent_part[1][0]

            # se il PoS è un verbo...
            if pos_tag == 'V':

                # ... controllo che il lemma corrisponda al verbo oggetto dell'analisi
                lemma = utils.get_lemma_from_word(current_sent_part[0].lower(), pos_tag)
                if lemma == CHOSEN_WORD:

                    # ottengo gli argomenti che si trovano subito a sinistra e destra del verbo
                    left_lemma = None
                    right_lemma = None

                    # prelevo il primo sostantivo andando verso sinistra
                    j = i - 1
                    while j >= 0:

                        left_pos_sent = tagged_sentence[j]
                        pos_tag = left_pos_sent[1][0]

                        if pos_tag == "N":
                            left_lemma = utils.get_lemma_from_word(left_pos_sent[0].lower(), pos_tag)
                            break
                        j -= 1

                    # prelevo il primo sostantivo andando verso destra
                    j = i + 1
                    while j < len(tagged_sentence):

                        right_pos_sent = tagged_sentence[j]
                        pos_tag = right_pos_sent[1][0]

                        if pos_tag == "N":
                            right_lemma = utils.get_lemma_from_word(right_pos_sent[0].lower(), pos_tag)
                            break
                        j += 1

                    # se ho trovato tutti e due i sostantivi...
                    if right_lemma is not None and left_lemma is not None:

                        # ... si ottengono i sensi dei due lemmi tramite l'algoritmo di Lesk
                        written_sentence = ' '.join(sentence.leaves())
                        left_sense = utils.simplified_lesk(written_sentence, left_lemma)
                        right_sense = utils.simplified_lesk(written_sentence, right_lemma)

                        # se ho trovato entrambi i sensi li aggiungo alla lista
                        if left_sense and right_sense:
                            verb_arguments_senses.append((left_sense, right_sense))

                    break

    # ora si passa all'individuazione dei supersensi e all'aggregazione dei risultati
    tot_sentences = len(verb_arguments_senses)

    # memorizzo i conteggi per ogni coppia di supersensi
    aggregated_supersense = defaultdict()

    for i in range(tot_sentences):
        arguments = verb_arguments_senses[i]

        # ottengo i supersensi
        # prendo il primo percorso che va dalla radice al senso
        # e prendo il secondo elemento di questo percorso che indica il supersenso
        left_path = arguments[0].hypernym_paths()[0]
        right_path = arguments[1].hypernym_paths()[0]

        # se il percorso è abbastanza lungo posso prendere il relativo supersenso
        if len(left_path) > SENSE_DEPTH and len(right_path) > SENSE_DEPTH:
            left_supersense = left_path[SENSE_DEPTH].name()
            right_supersense = right_path[SENSE_DEPTH].name()

            # aggiorno i conteggi per la coppia dei supersensi
            key = (left_supersense, right_supersense)
            if key in aggregated_supersense:
                aggregated_supersense[key] += 1
            else:
                aggregated_supersense[key] = 1

    # mostro i risultati
    print("Risultati dell'aggregazione dei supersensi su", tot_sentences, "frasi:")

    for k, v in sorted(aggregated_supersense.items(), key=lambda item: -item[1]):
        num = aggregated_supersense[k]
        print(num, "per la coppia ->", k[0], '+', k[1])

