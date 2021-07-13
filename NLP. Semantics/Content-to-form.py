import csv
from utils import get_lemmas_from_sentence, get_synset_context, simplified_lesk

csv_data_path = 'res_ex2/Esperimento content-to-form.csv'
NUMBER_OF_CANDIDATE_SYNSETS = 3

# Caricamento dei dati content-to-form
def read_csv_file():
    concept_definition = []
    with open(csv_data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            else:
                # Aggiunta alla lista concept_definition di un concetto con le definizioni ad esso associate
                concept_definition.append({
                    'concept': row[0],
                    'definitions': row[1:13]
                })
    return concept_definition


if __name__ == '__main__':
    concept_definition = read_csv_file()

    for concept in concept_definition:

        # Per ogni concetto a partire dalle sue definizioni creiamo un array contenente le frequenze dei termini che
        # occorrono più volte
        print("\nConcetto da individuare:", concept["concept"])
        print()

        # preprocessing delle definizioni
        lemma_sets = [get_lemmas_from_sentence(definition) for definition in concept["definitions"]]

        # calcolo delle occorrenze per ciascun lemma
        lemma_occurences = {}
        for lemma_set in lemma_sets:
            for lemma in lemma_set:
                if lemma in lemma_occurences:
                    lemma_occurences[lemma] += 1
                else:
                    lemma_occurences[lemma] = 1

        # Ordinamento decrescente dell'array delle frequenze dei termini in base alla frequenza
        sorted_lemma = sorted(lemma_occurences.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

        candidate_synset = []

        # Definizione del contesto associato ad un termine nei sorted lemma
        context = []

        # Per ogni lemma andiamo a ottenere le definizioni nelle quali esso compare, al fine di effettuare la WSD
        # per l'ottenimento del synset associato
        for index, lemma in enumerate(sorted_lemma):
            context.append('')

            # Loop sulle definizioni associate al concetto
            for definition in concept["definitions"]:
                if lemma[0] in get_lemmas_from_sentence(definition):
                    # Il contesto è dato dalle definizioni associate al concetto da individuare,
                    # all'interno delle quali è presente il termine che si sta analizzando
                    context[index] += ' ' + definition

            # WSD con Lesk del termine sfruttando il contesto individuato
            synset = simplified_lesk(context[index], lemma[0])
            if synset is not None:
                candidate_synset.append(synset)

            # Vengono considerati solo i primi tot. synset, dai quali poi si effetuerà il task content-to-form
            if len(candidate_synset) == NUMBER_OF_CANDIDATE_SYNSETS:
                break

        print("Synset candidati \n", candidate_synset, "\n")
        max = 0
        best_synset_in_subtree = None

        # Loop sui synset candidati
        for s in candidate_synset:

            # Insieme dei lemmi individuati a partire da tutte le definizioni del concetto da trovare
            concept_definitions_set = set.union(*lemma_sets)

            # Sottoalbero del synset in esame prendendo tutti i suoi iponimi
            hyponims_subtree = list(set(s.closure(lambda s: s.hyponyms())))

            for hyponim in hyponims_subtree:
                # ottengo il contesto del synset iponimo
                hyponim_cotext_set = get_synset_context(hyponim)

                # Calcolo similarità tra il contesto dell'iponimo e il contesto del concetto da trovare
                similarity = len(hyponim_cotext_set & concept_definitions_set) / (
                        min(len(hyponim_cotext_set), len(concept_definitions_set)) + 1)
                if similarity > max:
                    max = similarity

                    # Prendiamo come miglior synset corrispondente al concetto da trovare l'iponimo più simile
                    best_synset_in_subtree = hyponim

        print("Best Synset", best_synset_in_subtree)
