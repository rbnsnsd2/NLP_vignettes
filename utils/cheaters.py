import re
import numpy as np
from collections import Counter, defaultdict


def train_test_split(X, y, test_size=0.2):
    assert len(X) == len(y)
    x_len = len(X)
    samp_size = int(x_len*test_size)
    test_idx = np.random.choice(
        range(len(X)), size=samp_size, replace=False
    )
    train_idx = list(
        set(range(x_len)) - set(test_idx)
    )
    Xtrain = [X[i] for i in train_idx]
    ytrain = [y[i] for i in train_idx]
    Xtest = [X[i] for i in test_idx]
    ytest = [y[i] for i in test_idx]
    print('***train/test set created of length:{}/{}'.format(
        len(Xtrain), len(Xtest)))
    return Xtrain, Xtest, ytrain, ytest


class dctConstr():

    def __init__(self, **kwargs):
        self.tfidf_state = False
        self.UNKN = '\u22b9'
        self.specialchar = {
            'UNKN': u'\u22b9',
            'START': u'\u22b0',
            'END': u'\u22b1',
            'NUM': u'\u2203',
            'PAD': u'\u2200',
            'TELE': u'\u22a4',
            'BANK': u'\u22a5',
            'EMAIL': u'\u220a',
            'USER': u'\u22c3',
            'DATE': u'\u22fc',
            'TIME': u'\u2298',
            'CURRENCY': u'\u2201',
            'WEB': u'\u22d5'
        }

        self.terms = self.specialchar
        self.counts = Counter(list(self.specialchar.values()))
        self.num_terms = 1
        self.idf = {}
        kwargs = {**kwargs}

        if kwargs.get("stop_words"):
            self.stop_words = [term for term in kwargs["stop_words"]]
        else:
            self.stop_words = []

        if kwargs.get("ignore_case"):
            self.case = kwargs["ignore_case"]
        else:
            self.case = False

        if kwargs.get("char_level"):
            self.char_level = kwargs["char_level"]
        else:
            self.char_level = False

    def extract_entity(self, string):
        """
        replace key token sets with special characters
        e.g. emails, usernames, bank account numbers
        """
        _time = re.compile(" ?[\d]{1,2}\:[\d]{2} ?")
        _date = re.compile("(([12]\d{3}(//|-)(0[1-9]|1[0-2])(//|-)(0[1-9]|[12]\d|3[01]))|((0[1-9]|[12]\d|3[01])(\/|-)(0[1-9]|1[0-2])(\/|-)[12]\d{3}))")
        _user = re.compile(" ?[a-zA-Z0-9]{3,15}")
        _email = re.compile("[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
        _tele = re.compile("\+?\d[\d -]{8,12}\d")  # catch all...
        _bank = re.compile("[\W^][0-9]{10,16}[\W]")
        _num = re.compile("[\d]+")
        _currency = re.compile("[\u0024\u00a3\u00a5\u09f2\u09f3\u09fb\u0e3f\u0af1\u0bf9\u20a8\u20ac\u20b9]+")
        _web = re.compile("((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*")

        string = str(string)
        string = re.sub(_web, self.specialchar['WEB'], string)
        string = re.sub(_time, self.specialchar['TIME'], string)
        string = re.sub(_date, self.specialchar['DATE'], string)
        # string = re.sub(_user, self.specialchar['USER'], string)
        string = re.sub(_email, self.specialchar['EMAIL'], string)
        string = re.sub(_tele, self.specialchar['TELE'], string)
        string = re.sub(_bank, self.specialchar['BANK'], string)
        string = re.sub(_num, self.specialchar['NUM'], string)
        string = re.sub(_currency, self.specialchar['CURRENCY'], string)

        return string

    def sub_punctuation(self, document):
        doc = re.sub(r"[\!\_\.\n\'\:\;,\?]+", " ", document)
        return doc

    def segmenter(self, document):
        document = self.extract_entity(document)
        if self.case is True:
            document = document.lower()
        doc = self.sub_punctuation(document)
        seg = re.split(r"\s+", doc)
        seg = [i for i in seg if i]
        if self.stop_words:
            seg = self.remove_stop_words(seg)
        return seg

    def remove_stop_words(self, terms):
        """
        remove terms from list if in stop_words
        """
        _keepers = [t for t in terms if t not in self.stop_words]
        return _keepers

    def term_inator(self, _counter):
        """
        re-index terms-idx
        """
        _terms = {k: i for i, k in enumerate(_counter.keys(), start=0)}
        _indices = {k: i for i, k in _terms.items()}
        self.terms = _terms
        self.indices = _indices
        self.num_terms = len(_terms)

    def trimmer(self, **kwargs):
        """
        top refers to the most frequent percentage of terms
        bottom to the least frequent percentage of terms
        min is the lowest number of occurances to keep
        """
        kwargs = {**kwargs}
        _len = len(self.counts)
        
        if kwargs.get("max"):
            self.counts = Counter(
                el for el in self.counts.elements() if self.counts[el] <= kwargs["max"])
        if kwargs.get("min"):
            self.counts = Counter(
                el for el in self.counts.elements() if self.counts[el] >= kwargs["min"])

        print(f"before trim number of terms: {_len}")
        print(f"after trim: {len(self.counts)}")

        self.term_inator(self.counts)

    def constructor(self, document):
        """
        the constructor can be used iteratively such that the entire
        corpus doesnt have to be loaded into memory
        """
        if self.char_level is False:
            _terms = self.segmenter(document)
        else:
            _terms = document

        self.counts.update(_terms)
        self.term_inator(self.counts)

    def terms_to_idx(self, terms):
        cat_idx = [self.terms.get(term, 0) for term in terms]
        return cat_idx

    def terms_to_bow(self, terms):
        _idx = [self.terms.get(term, 0) for term in terms]
        _count = Counter(_idx)
        _bow = sorted([
            (idx, ct) for idx, ct in zip(_count.keys(), _count.values())
            ])
        return _bow

    def build_tfidf(self, corpus):
        """
        given the tokens and corpus, calc the
        inv freq of occurance of each token in
        each doc. only needed generate the idf
        for the classifier.
        """
        self.tfidf_state = True
        _docfreq = defaultdict(int)
        print('started building the idf')
        for chat in corpus:
            chat = self.segmenter(chat)
            for word, _ in self.terms_to_bow(chat):
                _docfreq[word] += 1
        if len(_docfreq) != self.num_terms:
            """
            if not all terms are in the dictionary, the idf value is zero
            hence the values are never considered in the models. So make a
            minimal correction that results in minimal value.
            """
            num_chats = len(corpus)
            for token_key in self.terms.values():
                if token_key not in _docfreq.keys():
                    # assume in every document
                    _docfreq[token_key] = num_chats
        self.idf = {
            word: np.log((len(corpus) + 1) / _docfreq[word])
            for word, freq in _docfreq.items()
        }

    def tfidf(self, document):
        _terms = self.segmenter(document)
        bow = self.terms_to_bow(_terms)
        word_count = sum(j for _, j in bow)
        out = [(i, (j/word_count)*self.idf.get(i, 0.001)) for i, j in bow]
        return out

    def bow_to_vec(self, bow):
        vec = np.zeros(self.num_terms + 1)
        for idx, _ct in bow:
            vec[idx] = _ct
        return vec

    def to_idx(self, document):
        _terms = self.segmenter(document)
        return self.terms_to_idx(_terms)

    def vec_to_terms(self, vec):
        _strings = [self.indices.get(item, 0) for item in vec]
        if self.char_level is True:
            _doc = "".join(_strings)
        else:
            _doc = " ".join(_strings)
        return _doc

    def __call__(self, document):
        if self.char_level is True:
            char_idx = [self.terms.get(term, 0) for term in document]
            return char_idx
        else:
            if self.tfidf_state:
                bow = self.tfidf(document)
                return bow
            else:
                _terms = self.segmenter(document)
                bow = self.terms_to_bow(_terms)
                return bow


def non_exclusive_chats(df):
    import pandas as pd
    from gensim import models, similarities

    dct = dctConstr(
        stop_words=["i", "you", "a", ""],
        ignore_case=True)

    print("constructing dictionary")
    for chat in df.chatline.tolist():
        dct.constructor(chat)
    dct.trimmer(min=5)

    print("converting corpus to bow")
    corp = df.chatline.tolist()
    bow = [dct(chat) for chat in corp]
    idx_to_terms = {i: j for j, i in dct.terms.items()}
    num_topics = 100
    print("Calculating LSA")
    lsa = models.LsiModel(bow, id2word=idx_to_terms, num_topics=num_topics)
    print("Transforming corpus into LSA-space")
    corpus_lsa = lsa[bow]
    print("Calculating the matrix similarities")
    index_lsa = similarities.MatrixSimilarity(corpus_lsa, num_best=2, num_features=dct.num_terms)

    print("Iterating through each of the chats, finding their NN and checking labels")
    exclusive_chats = []
    occluded_chats = []
    for i, l_chat in enumerate(corpus_lsa):
        nn, cosine = index_lsa[l_chat][-1]
        match = []
        if cosine >= 0.94:
            if i not in occluded_chats:
                if df.subtopic1.iloc[i] != df.subtopic1.iloc[nn]:
                    # print(f"{df.chatline.iloc[i]}::{df.subtopic1.iloc[i]}")
                    # print(f"{df.chatline.iloc[nn]}::{df.subtopic1.iloc[nn]}")
                    # print()
                    match.append(True)
                if df.topic1.iloc[i] != df.topic1.iloc[nn]:
                    # print(f"{df.chatline.iloc[i]}::{df.topic1.iloc[i]}")
                    # print(f"{df.chatline.iloc[nn]}::{df.topic1.iloc[nn]}")
                    # print()
                    match.append(True)
                if df.sentiment.iloc[i] != df.sentiment.iloc[nn]:
                    print(f"{df.chatline.iloc[i]}::{df.sentiment.iloc[i]}")
                    print(f"{df.chatline.iloc[nn]}::{df.sentiment.iloc[nn]}")
                    print()

                    match.append(True)
            if any(match):
                occluded_chats.extend([i, nn])
        else:
            exclusive_chats.append(i)

    _out = [df.chatline.iloc[i] for i in occluded_chats]
    df_out = pd.DataFrame(data={"chatline": _out,
                                "topic1": [""]*len(_out),
                                "subtopic1": [""]*len(_out),
                                "sentiment": [""]*len(_out)})
    df_out.to_csv("occluded_chats.csv", index=False)
    return f"Of the {len(df)} chatlines, {len(df_out)} were non-exclusive"


class modelBuilder:
    def __init__(self, dataframe, dict_parser):

        self.df = dataframe
        self.dct = dict_parser
        tfidf = [self.dct.tfidf(chat) for chat in self.df.chatline.tolist()]
        self.vec_t = [self.dct.bow_to_vec(p) for p in tfidf]

    def model_build(self,
                    df_label_name,
                    model_type,
                    train_test=False):
        from sklearn.metrics import confusion_matrix, classification_report
        self.model = model_type
        self.label_name = df_label_name
        self.labels = self.df[df_label_name].tolist()

        if train_test is False:
            self.model.fit(self.vec_t, self.labels)
            print('Accuracy score (no test data)\n{}: {}'.format(
                str(self.model), self.model.score(
                    self.vec_t, self.labels)))
            predictions = self.model.predict(self.vec_t)
            print(classification_report(self.labels, predictions))
            print(confusion_matrix(self.labels, predictions))
        else:
            self.train_test()
            self.model.fit(self.Xtrain, self.ytrain)
            print('Accuracy score on test data\n{}: {}'.format(
                str(self.model), self.model.score(
                    self.Xtest, self.ytest)))
            predictions = self.model.predict(self.Xtest)
            print(classification_report(self.ytest, predictions))
            print(confusion_matrix(self.ytest, predictions))

    def train_test(self, test_size=0.3):
        self.Xtrain, self.Xtest, self.ytrain, self.ytest =\
            train_test_split(self.vec_t,
                             self.labels,
                             test_size=test_size)

    def optimize_model(self, step=1, folds=2, scoring="accuracy"):
        from sklearn.feature_selection import RFECV
        from sklearn.model_selection import StratifiedKFold
        from matplotlib import pyplot as plt
        print("this is going to take a long time...")
        self.selector = RFECV(self.model,
                              step=step,  # features to remove at each step
                              cv=StratifiedKFold(folds),
                              verbose=1, scoring=scoring)
        self.selector.fit(self.vec_t, self.labels)
        print(f"optimal number of features: {self.selector.n_features_}")
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(self.selector.grid_scores_) + 1),
                 self.selector.grid_scores_)
        plt.show()

        # NEED A METHOD TO AUTOMATICALLY TRIM
        self.rank = sorted(
            [(i, j) for i, j in enumerate(self.selector.ranking_)], key=lambda x: x[1], reverse=True)

    def save_model(self, path):
        import pickle
        complete_path = "storage/"+path+".pkl"
        with open(complete_path, 'wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saved model to:{complete_path}")

def select_neutrals(dataframe, column_name, neutral, scaling):
    """
    since there are so many neutrals, we need to deal with the accuracy 
    paradox directly. Extract all non-neutrals, and add back no more 
    than 2x (scaling) as many neutrals with random selection...
    """
    from random import randrange
    def rand(exclude):
        r = None
        while r in [exclude, None, np.nan]:
            r = randrange(0, len(dataframe))
            return r

    pm_sent = []
    for i, item in enumerate(dataframe[column_name].values):
        if item in [neutral, np.nan]:
            pass
        else:
            pm_sent.append(i)

    pm_len = len(pm_sent)
    while len(pm_sent) <= (scaling * pm_len) + pm_len:
        pm_sent.append(rand(pm_sent))
    print(f'<<<without {neutral} there are {pm_len} records in df, \
{len(pm_sent)} after')
    return pm_sent


def df_selected(dfq, column_name, neutral, scaling=1):
    selected_lines = select_neutrals(dfq, column_name, neutral, scaling)
    return dfq.iloc[selected_lines]
