import nltk

def preprocess(file_name):
    # get data
    file = open(file_name, 'r')
    corpus = file.read()

    # tokenize data recieved
    tokenised_corpus = nltk.word_tokenize(corpus)

    # The freq dist of the tokenised corpus will be used to see which words are most common exclusively
    # in the pos and neg sets respectively
    # also gets rids of nouns (POS tags with NN) as they do not hold much sentimental value either
    # (do not need subjectivity)

    def skip_unwanted(pos_tuple):
        word, tag = pos_tuple
        return word.isalpha() and not tag.startswith("NN")

    # get rid of stop words in the set and only alphabets (no punctuation since we just ant the freq dis of the words
    tokenised_corpus = [word for word, tag in filter(
        skip_unwanted,
        nltk.pos_tag(tokenised_corpus)
    )]
    print(len(tokenised_corpus))

    freq_dist = nltk.FreqDist(tokenised_corpus)

    # get bigram frequencies as well
    bigram_list = nltk.collocations.BigramCollocationFinder.from_words(tokenised_corpus)

    # might remove trigrams as they do not offer much value, most of the common ones are just repeated movie names or
    # tag lines and cannot be generalised to actually predict the sentiment

    return freq_dist, bigram_list.ngram_fd
