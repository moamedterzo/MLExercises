from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# dizionario che mappa il corretto PoS per il lemmatizer seguente
tag_dict = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}


# preprocessing della frase che restituisce i lemmi, rimuovendo stop words
def get_lemmas_from_sentence(sentence):

    tokens = word_tokenize(sentence.lower())

    result = []

    # determinazione PoS
    pos_tags = pos_tag(tokens)

    for i in range(len(pos_tags)):
        # ottenimento tag
        tag = pos_tags[i]

        lemma = get_lemma_from_word(tokens[i], tag[1][0].upper())

        if lemma not in stop_words and lemma.isalnum():
            result.append(lemma)

    return set(result)


# effettua un conteggio dei lemmi dei token
def get_lemmas_from_tokens_as_counts(tokens):
    result = {}

    # determinazione PoS
    pos_tags = pos_tag(tokens)

    for i in range(len(pos_tags)):
        # ottenimento tag
        tag = pos_tags[i]

        lemma = get_lemma_from_word(tokens[i], tag[1][0].upper())

        if lemma not in stop_words and lemma.isalnum():
            if lemma in result:
                result[lemma] += 1
            else:
                result[lemma] = 1

    return result


# dalla singola parola restituisce il lemma
def get_lemma_from_word(word, PoS):
    wn_pos_tag = tag_dict.get(PoS, wordnet.NOUN)

    # lemmatizzazione
    return lemmatizer.lemmatize(word, wn_pos_tag)


# si ottiene il contesto del synset dato dalla sua definizione e dai suoi iponimi e iperonimi
def get_synset_context(synset, compute_related=False):
    context = set()

    context |= get_lemmas_from_sentence(synset.definition())

    for example in synset.examples():
        context |= get_lemmas_from_sentence(example)

    if compute_related:
        for hypernym in synset.hypernyms():
            context |= get_synset_context(hypernym)

        for hyponym in synset.hyponyms():
            context |= get_synset_context(hyponym)

    return context


# implementazione dell'algoritmo Lesk
def simplified_lesk(sentence, word_to_disambiguate, word_pos_tag=wordnet.NOUN):

    # ottengo l'insieme dei lemmi per la frase
    context = get_lemmas_from_sentence(sentence)

    # ottengo i synset candidati per la parola da disambiguare
    candidate_synsets = wordnet.synsets(word_to_disambiguate, pos=word_pos_tag)

    if not candidate_synsets:
        return None

    # si restituisce il synset che possiede maggiori termini in comune col contesto della frase
    best_sense = None
    max_overlap = -1

    for sense in candidate_synsets:
        # ottengo il contesto del synset (vedi definizione metodo)
        signature = get_synset_context(sense)

        # si calcola lo score
        overlap = len(signature.intersection(context))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense

    return best_sense