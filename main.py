import numpy
import pandas as pd


def write_outputs(input, flag):
    if flag == 1:
        out = 'out1.txt'
    else:
        out = 'out2.txt'
    with open(out, mode="w") as outfile:
        for item in input:
            outfile.write("{}\n".format(item))


def read_txt():
    all_data = []
    dataset_list = []
    file_1 = "toy-en.txt"
    file_2 = "toy-fa.txt"

    dataset_list.append(file_1)
    dataset_list.append(file_2)

    for i in range(len(dataset_list)):
        # az dataset_list khuneh i om file txt ro bekhun be onvane file
        with open(dataset_list[i], 'r', encoding='UTF-8') as file:
            text = file.read()
            all_data.append(text)
    return all_data


def wrapp_list_as_dict(list_data):
    corp = []
    _l1 = list_data[0].splitlines()
    _l2 = list_data[1].splitlines()
    for i in range(len(_l1)):
        corp.append({'fr': _l2[i], 'en': _l1[i]})
    return corp


def get_words(corpus):
    def source_words(lang):
        for pair in corpus:
            for word in pair[lang].split():
                yield word

    return {lang: set(source_words(lang)) for lang in ('en', 'fr')}


def init_probabilities(corpus):
    init_holder = []
    for (es, fs) in [(pair['en'].split(), pair['fr'].split())
                     for pair in corpus]:
        init_prob = pd.DataFrame(1, columns=es, index=fs)
        init_holder.append(init_prob)
        init_prob = []

    return init_holder


def train_iteration(corpus, words, _probabilities):
    counts = {word_en: {word_fr: 0 for word_fr in words['fr']}
              for word_en in words['en']}

    counts_2 = {word_en: {word_fr: 0 for word_fr in words['en']}
                for word_en in words['fr']}

    maximization_matrix = pd.DataFrame(0, index=counts_2, columns=counts)

    memory = []
    for (es, fs) in [(pair['en'].split(), pair['fr'].split())
                     for pair in corpus]:

        for i, e in enumerate(es):
            for j, f in enumerate(fs):
                if not any(e == sub_memory[0] and f == sub_memory[1] or f == sub_memory[0] and e == sub_memory[
                    1] in sub_memory for sub_memory in memory):
                    for k in range(len(_probabilities)):
                        current_mini_mat = _probabilities[k]
                        try:
                            if current_mini_mat.loc[f, e]:
                                maximization_matrix.loc[f, e] += current_mini_mat.loc[f, e]
                        except KeyError:
                            pass
                else:
                    pass
                memory.append([e, f])

    for r in range(numpy.size(maximization_matrix, 0)):
        maximization_matrix.iloc[r, :] = maximization_matrix.iloc[r, :] / sum(maximization_matrix.iloc[r, :])
    maximization_matrix = maximization_matrix.fillna(0)

    for i, row in maximization_matrix.iterrows():
        for j, col in maximization_matrix.iteritems():
            for k in range(len(_probabilities)):
                for v, row_ind in _probabilities[k].iterrows():
                    for b, col_ind in _probabilities[k].iteritems():
                        try:
                            if i == v and j == b:
                                _probabilities[k].loc[i, j] = maximization_matrix.loc[i, j]
                        except KeyError:
                            pass

    for k in range(len(_probabilities)):
        for b, col_ind in _probabilities[k].iteritems():
            _probabilities[k].loc[:, b] = _probabilities[k].loc[:, b] / sum(_probabilities[k].loc[:, b])

    return _probabilities, maximization_matrix


def EM(corpus):
    words = get_words(corpus)

    init_prob = init_probabilities(corpus)

    iterations = 5
    for iter in range(iterations):
        translation_probabilities, max_mat = train_iteration(corpus, words, init_prob)
        init_prob = translation_probabilities
    return translation_probabilities, max_mat


def print_save_results(max_mat):
    #print title
    # print("################################output 1 ####################################")
    # print("{:<8} {:<15} {:<10}".format('English word', 'Farsi word', 'probability'))
    en_list = list(max_mat.columns.values)
    fa_list = list(max_mat.index.values)
    #
    out1 = []
    for i, en_val in enumerate(en_list):
        for j, fa_val in enumerate(fa_list):
            print("{:<8} {:<15} {:<10}".format(en_val, fa_val, max_mat.iloc[i, j]))
            out1.append([en_val, fa_val, max_mat.iloc[i, j]])

    write_outputs(out1, 1)

    out2 = []
    print("################################output 2 ####################################")
    for k, col_name in enumerate(fa_list):
        maxValueIndexObj = max_mat.idxmax(axis=1)
        print(maxValueIndexObj[k], col_name, max(max_mat.iloc[k, :]))
        out2.append([maxValueIndexObj[k], col_name, max(max_mat.iloc[k, :])])

    write_outputs(out2, 2)

    return en_list, fa_list


def main():
    mydata = read_txt()
    corpus = wrapp_list_as_dict(mydata)
    probabilities, max_mat = EM(corpus)
    first_part, second_part = print_save_results(max_mat)
    first_part = pd.DataFrame(first_part)
    second_part = pd.DataFrame(second_part)
    max_mat.to_csv('final_dict.csv', sep=',', index=None, header=None)
    first_part.to_csv('first_part.csv', sep=',', index=None, header=None)
    second_part.to_csv('second_part.csv', sep=',', index=None, header=None)


if __name__ == '__main__':
    main()
