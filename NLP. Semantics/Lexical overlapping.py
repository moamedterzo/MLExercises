import xlrd
import statistics
from utils import get_lemmas_from_sentence

excel_data_path = 'res_ex1/definizioni.xlsx'


def read_excel_file():
    # apro il file excel
    wb = xlrd.open_workbook(excel_data_path)
    sheet = wb.sheet_by_index(0)

    # ci sono 2 proprietà da associare al concetto che sono definire a priori nell'intestazione
    concepts = []

    for col in range(1, 5):
        header = sheet.cell_value(0, col)

        header_data = header.replace("Definizione ", "").split("_")
        definitions = get_concept_definitions(col, sheet)

        concepts.append({
            'concept': header_data[2],
            'dim1': header_data[0],
            'dim2': header_data[1],
            'definitions': definitions
        })

    return concepts


# Restituisce le definizioni per un concetto specifico nel file excel
def get_concept_definitions(col, sheet):

    definitions = []
    for i in range(1, 20):
        definition = sheet.cell_value(i, col)
        if definition:
            definitions.append(definition)

    return definitions


def compute_definitions_similarity(definitions):

    # dalle definizioni ottengo delle parole preprocessate
    prep_definitions = [get_lemmas_from_sentence(definition) for definition in definitions]

    # la similarità tra tutte le definizioni è calcolata come la media tra tutti i valori di similarità
    # di ciascuna possibile coppia di definizioni
    similarities = []

    for i in range(len(prep_definitions) - 1):
        for j in range(i + 1, len(prep_definitions)):
            # la similarità tra due definizioni è data dall'intersezione delle lemmi fratto la cardinalità dell'insieme più piccolo
            intersection = len(prep_definitions[i] & prep_definitions[j])

            # definizione avente cardinalità minima
            min_card_definition = min(len(prep_definitions[i]), len(prep_definitions[j]))

            similarities.append(float(intersection) * 100 / min_card_definition)

    return statistics.mean(similarities)


if __name__ == '__main__':

    data = read_excel_file()

    dim1_similarities = {}
    dim2_similarities = {}

    for item in data:
        # calcolo similarità
        similarity = compute_definitions_similarity(item['definitions'])

        dim1 = item['dim1']
        dim2 = item['dim2']

        # aggiunta valori della similarità per le due dimensioni
        if dim1 not in dim1_similarities:
            dim1_similarities[dim1] = []
        dim1_similarities[dim1].append(similarity)

        if dim2 not in dim2_similarities:
            dim2_similarities[dim2] = []
        dim2_similarities[dim2].append(similarity)

        print("Similarity for definitions of concept {}: {}%".format(item['concept'], int(similarity)))

    # calcolo della similarità aggregata per le due dimensioni
    print()
    for dim1 in dim1_similarities:
        dim_similarities = dim1_similarities[dim1]
        mean_similarity = statistics.mean(dim_similarities)
        print("Mean similarity for '{}' concepts: {}%".format(dim1, int(mean_similarity)))

    print()
    for dim2 in dim2_similarities:
        dim_similarities = dim2_similarities[dim2]
        mean_similarity = statistics.mean(dim_similarities)
        print("Mean similarity for '{}' concepts: {}%".format(dim2, int(mean_similarity)))
