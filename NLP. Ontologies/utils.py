from scipy.stats import spearmanr, pearsonr

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


def get_lemmas_from_sentence(sentence, get_only_lemmas=True, word_to_disambiguate=None):

    tokens = word_tokenize(sentence.lower())

    # se la parola da disambiguare non c'è, la devo prelevare
    special_word_index = None
    if not get_only_lemmas:
        for i in range(len(tokens)):
            token = tokens[i]
            # ottenimento indice parola da disambiguare
            if token == word_to_disambiguate:
                special_word_index = i
                break
            elif token.startswith("*"):
                tokens[i] = token.strip('*')
                special_word_index = i
                break

    result = []

    # determinazione PoS
    pos_tags = pos_tag(tokens)

    word_pos_tag = None

    for i in range(len(pos_tags)):
        # ottenimento tag
        tag = pos_tags[i]
        wn_pos_tag = tag_dict.get(tag[1][0].upper(), wordnet.NOUN)

        # lemmatizzazione
        lemma = lemmatizer.lemmatize(tokens[i], wn_pos_tag)

        # memorizzazione lemma parola e pos tag
        if not get_only_lemmas and i == special_word_index:
            if word_to_disambiguate is None:
                word_to_disambiguate = lemma
            word_pos_tag = wn_pos_tag

        if lemma not in stop_words and lemma.isalnum():
            result.append(lemma)

    if get_only_lemmas:
        return set(result)
    else:
        return set(result), word_to_disambiguate, word_pos_tag


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


def pearson_coefficient(a, b):
    # calculate Pearson's correlation
    corr, _ = pearsonr(a, b)
    print('Pearson''s correlation: %.3f' % corr)


def spearman_coefficient(a,b):

    # calculate Spearman's correlation
    coef, p = spearmanr(a, b)
    print('Spearman''s correlation coefficient: %.3f' % coef)