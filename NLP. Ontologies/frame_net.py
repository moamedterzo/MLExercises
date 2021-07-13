from random import randint,seed
import json
import math
from utils import get_lemmas_from_sentence, get_synset_context

import nltk
from nltk.corpus import framenet as fn
from nltk.corpus import wordnet as wn


# region Metodi Prof.
def print_frames_with_IDs():
    for x in fn.frames():
        print('{}\t{}'.format(x.ID, x.name))


def get_frams_IDs():
    return [f.ID for f in fn.frames()]


def getFrameSetForStudent(surname, list_len=5):
    nof_frames = len(fn.frames())
    base_idx = (abs(hash(surname)) % nof_frames)
    print('\nstudent: ' + surname)
    framenet_IDs = get_frams_IDs()
    i = 0
    offset = 0
    seed(1)
    while i < list_len:
        fID = framenet_IDs[(base_idx + offset) % nof_frames]
        f = fn.frame(fID)
        fNAME = f.name
        print('\tID: {a:4d}\tframe: {framename}'.format(a=fID, framename=fNAME))
        offset = randint(0, nof_frames)
        i += 1


# getFrameSetForStudent('Racanati')
# getFrameSetForStudent('Sansonetti')

'''
student: Racanati
	ID: 2673	frame: Being_incarcerated
	ID:  361	frame: Emanating
	ID:  226	frame: Asymmetric_reciprocality
	ID:  418	frame: Collaboration
	ID:  950	frame: Law

student: Sansonetti
	ID: 1816	frame: Removing_scenario
	ID:  158	frame: Accoutrements
	ID:  513	frame: Purpose
	ID: 1130	frame: Studying
	ID: 2502	frame: Cognitive_impact
'''

#endregion

# lunghezza massima del cammino utilizzato nel mapping a grafi
LONGEST_GRAPH_PATH = 3
# 3)    86 correct synsets over a total of 181 -> 47.51381215469613 %
# 4, 5) 81 correct synsets
# 6)    83 correct synsets over a total of 181 -> 45.85635359116022 %
# 7)    84 correct synsets over a total of 181 -> 46.408839779005525 %
# 8)    85 correct synsets over a total of 181 -> 46.96132596685083 %
# 9)    83 correct synsets over a total of 181 -> 45.85635359116022 %
# 10)   83 correct synsets over a total of 181 -> 45.85635359116022 %
# 14)   83 correct synsets over a total of 181 -> 45.85635359116022 %


def elaborate_frame(frame_id, skip_frame_annotation=False):
    f = fn.frame(frame_id)
    print("Frame name:", f.name)

    # ottengo i synset candidati per le varie parti del frame
    f_data_synsets = {
            'name': get_synsets_for_frame_name(f.name),
            'LUs': get_synsets_for_lexical_units(f.lexUnit),
            'FEs': get_synsets_for_frame_elements(f.FE)
        }

    # ottengo le annotazioni
    file = None
    try:
        file = open("frame_annotations\\" + f.name + ".txt")
    except IOError:
        pass

    # se le annotazioni non sono presenti o le si vogliono effettuare nuovamente...
    if file is None or (skip_frame_annotation is False and input("Do you want to make annotation (y/n)?") == "y"):
        f_annotations = make_annotations(f, f_data_synsets)

        # memorizzo le annotazioni
        file = open("frame_annotations\\" + f.name + ".txt", 'w')
        json.dump(f_annotations, file)
    else:
        # carico le annotazioni
        f_annotations = json.load(file)

    # si ottiene il contesto del frame
    # N.B. il contesto viene utilizzato per mappare i sensi ai vari elementi del Frame
    # il contesto del frame sempre lo stesso
    frame_context = get_frame_context(f)

    # si utilizzano i due metodi di mapping dei termini ai synset
    result_bow = compute_mapping_bag_of_words(f_data_synsets, frame_context)
    result_graph = compute_mapping_graph(f_data_synsets, frame_context)

    # si ottengono i risultati di accuratezza
    total, correct_bow = compare_results(result_bow, f_annotations, f_data_synsets)
    _, correct_graph = compare_results(result_graph, f_annotations, f_data_synsets)

    print(total, "terms, correct for", "bow:", correct_bow, ", for graph:", correct_graph)
    print("-------------------")

    return total, correct_bow, correct_graph


# metodo che restituisce i synset associati a ciascuna lexical unit
def get_synsets_for_lexical_units(lex_units):
    result = {}
    for lu in lex_units:
        lu_parts = lu.split('.')

        # ottengo il PoS per Wordnet
        pos = lu_parts[1]
        if pos == 'adv':
            pos = 'r'
        elif pos not in ('a', 'v', 'n'):
            pos = None

        # si ottengono i synset dalla parola
        word = lu_parts[0]

        lu_result = []
        for synset in wn.synsets(word, pos=pos):
            lu_result.append(synset)
        if not lu_result:
            # se non si trova nulla si prova senza il PoS
            for synset in wn.synsets(word):
                lu_result.append(synset)

        result[word] = lu_result

    return result


# metodo che restituisce i synset associati a ciascun Frame Element
def get_synsets_for_frame_elements(FEs):
    result = {}
    for fe in FEs:

        fe_result = []
        for synset in wn.synsets(fe):
            fe_result.append(synset)

        result[fe] = fe_result

    return result


# preleva i synset dal nome del frame
def get_synsets_for_frame_name(frame_name):

    phrase = frame_name.replace("_", " ")
    synsets = wn.synsets(phrase)

    if synsets:
        return {phrase: synsets}
    else:
        # se non è possibile ottenere i synset dal nome composto, lo si prende dal reggente
        tags = nltk.pos_tag(frame_name.split("_"))

        # dato che la casistica è semplice, sappiamo già cosa prendere a seconda dei casi
        # caso in cui sono presenti due verbi alla fine, si prende l'ultimo verbo
        if tags[-1][1].startswith("VB") and tags[-2][1].startswith("VB"):
            word = tags[-1][0]
        # verbo+nome, si prende il verbo
        elif tags[-1][1].startswith("NN") and tags[-2][1].startswith("VB"):
            word = tags[-2][0]
        else:
            # di defult si prende sempre l'ultima parola
            word = tags[-1][0]

        return {word: wn.synsets(word)}


# il contesto del frame è dato dalle parole che occorrono nella sua definizione
# e nelle definizioni dei suoi Frame Elements
def get_frame_context(frame):
    context = set()

    # definizione del frame
    context |= get_lemmas_from_sentence(frame.definition)

    # definizione dei FE
    for fe in frame.FE:
        context |= get_lemmas_from_sentence(frame.FE[fe].definition.replace("This FE identifies", ""))

    return context


# ottiene il mapping degli elementi del frame utilizzando l'approccio Bag of Words
def compute_mapping_bag_of_words(frame_data, frame_context):
    result = {}

    # mapping del senso per il Frame name
    result['name'] = {}
    for n in frame_data['name']:
        result['name'][n] = get_best_mapping_bag_of_words(frame_context, frame_data['name'][n])

    # mapping del senso per ciascuna LU
    result['LUs'] = {}
    for lu in frame_data['LUs']:
        result['LUs'][lu] = get_best_mapping_bag_of_words(frame_context, frame_data['LUs'][lu])

    # mapping del senso per ciascun FE
    result['FEs'] = {}
    for fe in frame_data['FEs']:
        result['FEs'][fe] = get_best_mapping_bag_of_words(frame_context, frame_data['FEs'][fe])

    return result


# implementa l'approccio di mapping Bag of Words
def get_best_mapping_bag_of_words(frame_context, synsets):

    # casi aventi zero o un solo synset
    if synsets is None or len(synsets) == 0:
        return None
    elif len(synsets) == 1:
        return synsets[0]

    # ottenimento del synset che massimizza lo score
    max_score = 0
    best_synset = None

    for synset in synsets:
        synset_context = get_synset_context(synset, True)

        # lo score è dato dalla cardinalità dell'intersezione dei due contesti, più 1
        score = len(frame_context & synset_context) + 1

        if score > max_score:
            max_score = score
            best_synset = synset

    return best_synset


# ottiene il mapping degli elementi del frame utilizzando l'approccio a Grafo
def compute_mapping_graph(frame_data, frame_context):
    result = {}

    # ottengo l'insieme dei synset associati a ciascuna parola del contesto del frame
    frame_context_synsets = []
    for term in frame_context:
        frame_context_synsets.extend(wn.synsets(term))

    # mapping del senso per il Frame name
    result['name'] = {}
    for n in frame_data['name']:
        result['name'][n] = get_best_mapping_graph(frame_context_synsets, n, frame_data['name'][n])

    # mapping del senso per ciascuna LU
    result['LUs'] = {}
    for lu in frame_data['LUs']:
        result['LUs'][lu] = get_best_mapping_graph(frame_context_synsets, lu, frame_data['LUs'][lu])

    # mapping del senso per ciascun FE
    result['FEs'] = {}
    for fe in frame_data['FEs']:
        result['FEs'][fe] = get_best_mapping_graph(frame_context_synsets, fe, frame_data['FEs'][fe])

    return result


# implementa l'approccio di mapping a Grafo
def get_best_mapping_graph(frame_context_synsets, word, candidate_synsets):

    # caso in cui sono presenti zero o un synset candidato
    if candidate_synsets is None or len(candidate_synsets) == 0:
        return None
    elif len(candidate_synsets) == 1:
        return candidate_synsets[0]

    # calcolo dello score
    # visto che nella formula (data nel pdf) il denominatore non è influenzato dal senso che si sta scegliendo ma solo
    # dalla parola da mappare, e visto che l'obiettivo è di trovare il senso che massimizza la formula;
    # si può scartare il denominatore massimizzando lo score al numeratore
    max_score = -1
    best_synset = None

    for s in candidate_synsets:

        score = 0
        for context_synset in frame_context_synsets:
            # calcolo dei percorsi tra il synset candidato e quello del contesto
            paths = compute_possibile_paths(s, context_synset)

            for path in paths:
                # incremento dello score (sommatoria per tutti i synset del contesto e tutti i percorsi trovati)
                score += math.exp(-(path - 1))

        if score > max_score:
            max_score = score
            best_synset = s

    #if best_synset is not None and len(candidate_synsets) > 1:
    #    print("Best synset for ", word, " is ", best_synset, " with definition ", best_synset.definition())

    return best_synset


# calcola tutti i possibili percorsi tra due synset in WordNet
def compute_possibile_paths(s1, s2):

    # insieme delle lunghezze dei vari percorsi
    paths = []

    if s1 == s2:
        # percorso di lunghezza zero
        paths.append(0)

    # liste di frontiera che contengono gli iperonimi dei due synset iniziali
    list_s1 = []
    list_s2 = []

    list_s1.append((s1,))
    list_s2.append((s2,))

    max_path_length = 1

    # si termina quando la lunghezza dei percorsi trovati supera il limite massimo
    while max_path_length <= LONGEST_GRAPH_PATH:
        # calcolo i nodi iperonimi dello strato precedente
        hypernyms_s1 = []
        for el in list_s1[-1]:
            hypernyms_s1.extend(el.hypernyms())
        list_s1.append(hypernyms_s1)

        # controllo se gli iperonimi dell'ultimo strato sono presenti nella seconda lista
        for i in range(len(list_s2)):
            # lunghezza del percorso (non deve superare il limite)
            path_length = len(list_s1) - 1 + i
            if path_length > LONGEST_GRAPH_PATH:
                break

            # se un elemento nella frontiera è presente anche in uno strato della seconda lista, allora esiste un
            # percorso che collega i due synset e ha una lunghezza calcolabile
            for s2_strate_element in list_s2[i]:
                for frontier_element in list_s1[-1]:
                    if frontier_element == s2_strate_element:
                        paths.append(path_length)

        # espando lo strato della seconda lista
        # ed effettuo gli stessi controlli in maniera inversa
        hypernyms_s2 = []
        for el in list_s2[-1]:
            hypernyms_s2.extend(el.hypernyms())
        list_s2.append(hypernyms_s2)

        # controllo se gli iperonimi dell'ultimo strato sono presenti nella prima lista
        for i in range(len(list_s1)):
            # lunghezza del percorso (non deve superare il limite)
            path_length = len(list_s2) - 1 + i
            if path_length > LONGEST_GRAPH_PATH:
                continue

            # se un elemento nella frontiera è presente anche in uno strato della prima lista, allora esiste un
            # percorso che collega i due synset e ha una lunghezza calcolabile
            for s1_strate_element in list_s1[i]:
                for frontier_element in list_s2[-1]:
                    if frontier_element == s1_strate_element:
                        paths.append(path_length)

        max_path_length += 1

    return paths


# calcola il totale dei termini correttamente mappati
def compare_results(result, annotations, f_data_synsets):

    # i termini aventi zero o un solo synset candidato vengono scartati dai conteggi, in quanto sovrastimerebbero
    # l'indice di accuratezza
    total = 0
    correct = 0

    # senso del nome del frame
    for n in result['name']:
        total+=1
        if result['name'][n].name() == annotations['name'][n]:
            correct+=1

    # senso per ciascuna LU
    for lu in result['LUs']:
        if f_data_synsets['LUs'][lu] is not None and len(f_data_synsets['LUs'][lu]) > 1:
            total += 1
            if result['LUs'][lu].name() == annotations['LUs'][lu]:
                correct += 1

    # senso per ciascun FE
    for fe in result['FEs']:
        if f_data_synsets['FEs'][fe] is not None and len(f_data_synsets['FEs'][fe]) > 1:
            total += 1
            if result['FEs'][fe].name() == annotations['FEs'][fe]:
                correct += 1

    return total, correct


# metodo che permette di effettuare l'annotazione del frame in maniera più velocemente
def make_annotations(f, f_data_synsets):

    print('MAKING ANNOTATIONS')
    print("Synsets for frame name = " + f.name)
    print(f.definition)

    result = {}

    # nome del frame
    result['name'] = {}
    for n in f_data_synsets['name']:
        result['name'][n] = chose_synset_for_annotation(f_data_synsets['name'][n])
    print()


    # lexical units
    result['LUs'] = {}
    for lu in f.lexUnit:
        print("Synsets for lexical unit = " + lu)
        lu_name = lu.split('.')[0]
        result['LUs'][lu_name] = chose_synset_for_annotation(f_data_synsets['LUs'][lu_name])
        print()

    # frame elements
    result['FEs'] = {}
    for fe in f.FE:
        print("Synsets for frame element = " + fe)
        print(f.FE[fe].definition)
        result['FEs'][fe] = chose_synset_for_annotation(f_data_synsets['FEs'][fe])
        print()

    return result


def chose_synset_for_annotation(candidate_synsets):
    if candidate_synsets is None or len(candidate_synsets) == 0:
        return None

    for i in range(len(candidate_synsets)):
        synset = candidate_synsets[i]
        print(str(i) + ": ", synset, ". Definition: " + synset.definition())

    if len(candidate_synsets) == 1:
        return candidate_synsets[0].name()

    chosen = -1
    while chosen < 0 or chosen >= len(candidate_synsets):
        chosen = int(input("Choose a synset:"))

    return candidate_synsets[chosen].name()


if __name__ == '__main__':

    # elenco degli ID dei frame selezionati
    frames = [2673, 361, 226, 418, 950, 1816, 158, 513, 1130, 2502]

    # si memorizzano i totali corretti
    total = 0
    correct_graph = 0
    correct_bow = 0

    for frame in frames:
        sub_total, sub_correct_bow, sub_correct_graph = elaborate_frame(frame, skip_frame_annotation=True)

        total+= sub_total
        correct_bow+= sub_correct_bow
        correct_graph+= sub_correct_graph

    # mostro indici accuratezza
    print("Accuracy of Bag of words:")
    print(correct_bow, "correct synsets over a total of", total, "->", correct_bow*100/total,"%")

    print("Accuracy of Graph with length "+ str(LONGEST_GRAPH_PATH) +":")
    print(correct_graph, "correct synsets over a total of", total,"->", correct_graph*100/total,"%")
