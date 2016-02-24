import codecs
from nltk import word_tokenize
from operator import itemgetter
from collections import defaultdict
import os


class IBMModel1:
    """
    This class implements the IBM Model 1, an algorithm for word-alignment given
    a sentence-aligned corpus.
    It will load a corpus and store the probability of translating a given foreign word
    f into an english word e into a dictionary.
    """

    def __init__(self, num_sentences=20000):
        self.corpus = self.initialize_corpus(num_sentences)
        self.trans_prob = self.initialize_translation_probability()
        self.conditional_dict = defaultdict(list)

    def initialize_corpus(self, num_sentences=10000):
        """
        :param num_sentences: the default number of sentences loaded from a corpus
        :return: a dictionary of English-Spanish sentences, tokenized and stored as tuples of strings
        This method accepts the absolute path of the directory in which the aligned corpus
        files are located and loads them into a dictionary to be used for training the model.
        """
        directory_name = input("Enter a directory name to be read: \n")
        file_list = []
        for file in os.listdir(directory_name):
            if file.endswith("en"):
                filename = directory_name + file
                file_list.insert(0, (codecs.open(filename, "r", "utf-8")))
            elif file.endswith("es"):
                filename = directory_name + file
                file_list.insert(1, (codecs.open(filename, "r", "utf-8")))
        i = 0
        corpus = dict()
        while i < num_sentences:
            sentence1 = tuple(word_tokenize("NULL " + file_list[0].readline().strip("\n").strip("¡").strip("¿").lower()))
            sentence2 = tuple(word_tokenize("NULL " + file_list[1].readline().strip("\n").strip("¡").strip("¿").lower()))
            corpus[sentence1] = sentence2
            i += 1
        return corpus

    def initialize_translation_probability(self):
        """
        Initializes the initial probability of t(e|f) to the number of distinct word types in the foreign corpus
        :return: a default dictionary whose default value is 1/number of foreign words.
        """
        num_f_words = len(set(f_word for (english_sent, foreign_sent) in self.corpus.items() for f_word in foreign_sent))
        trans_prob = defaultdict(lambda: float(1/num_f_words))
        return trans_prob

    def train_model(self, iteration_count=100):
        """
        Iterates through the model for a set number of times, updating the t(e|f) probabilities by
        gathering evidence using counts.
        :return: the translation probability t(e|f) for lexical items in a given sentence
        """
        for i in range(iteration_count):
            count_e_given_f = defaultdict(float)
            total = defaultdict(float)
            sentence_total = defaultdict(float)
            for (english_sent, foreign_sent) in self.corpus.items():
                for e_word in english_sent:
                    for f_word in foreign_sent:
                        sentence_total[e_word] += self.trans_prob[(e_word, f_word)]
                for e_word in english_sent:
                    for f_word in foreign_sent:
                        count_e_given_f[(e_word, f_word)] += (self.trans_prob[(e_word, f_word)]/sentence_total[e_word])
                        total[f_word] += (self.trans_prob[(e_word, f_word)]/sentence_total[e_word])
            for (e_word, f_word) in count_e_given_f:
                self.trans_prob[(e_word, f_word)] = count_e_given_f[(e_word, f_word)]/total[f_word]

        return self.trans_prob

    def cond_dict(self):
        """
        Creates a conditional dictionary which can be accessed using the f_words as keys.
        :return: A dictionary that stores all possible translations as tuples of (e_word, value)
        """
        for ((e_word, f_word), value) in self.trans_prob.items():
            self.conditional_dict[f_word].append((e_word, value))

    def get_max(self, f_word, cond_dict):
        """
        Using a conditional dictionary,
        :param f_word: a foreign word, cond_dict -- a conditional dictionary from .cond_dict()
        :return: the max (e_word, value), i.e. the highest probability e translation along with its probability
        """
        maxi = (None, 0)
        for tuple in self.conditional_dict[f_word]:
            if tuple[1] > maxi[1]:
                maxi = tuple
        return maxi

    def print(self, num_iterations=100):
        """
        Prints the values of t(e|f) for a specified number of items in the dictionary.
        """
        print("Lexical alignment probabilities")
        print()
        print("{:<40}{:>40}".format("t(e|f)", "Value"))
        print("--------------------------------------------------------------------------------")
        iterations = 0
        for ((e_word, f_word), value) in sorted(self.trans_prob.items(), key=itemgetter(1), reverse=True):
            if iterations < num_iterations:
                print("{:<40}{:>40.2}".format("t(%s|%s)" % (e_word, f_word), value))
            else:
                break
            iterations += 1


if __name__ == "__main__":
    ibm = IBMModel1()
    ibm.train_model()
    ibm.print()
