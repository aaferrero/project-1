import sys
from gensim import matutils
from gensim.models import Word2Vec
import pickle
import scipy
import numpy as np
from gensim import corpora, models
import numpy as np
from sklearn.decomposition import PCA
from typing import List


class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector

        # a sentence, a list of words


class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list

        # return the length of a sentence

    def len(self) -> int:
        return len(self.word_list)

        # convert a list of sentence with word2vec items into a set of sentence vectors


def sentence2vec(wdfs, token2id, sentenceList: List[Sentence], embeddingSize: int, charLen: int, a: float = 1e-3):
    sentenceSet = []
    for sentence in sentenceList:
        sentenceVector = np.zeros(embeddingSize)
        for word in sentence.word_list:
            p = wdfs[token2id[word.text]] / charLen
            a = a / (a + p)
            sentenceVector = np.add(sentenceVector, np.multiply(a, word.vector))
        sentenceVector = np.divide(sentenceVector, sentence.len())
        sentenceSet.append(sentenceVector)
        # caculate the PCA of sentenceSet
    pca = PCA(n_components=embeddingSize)
    pca.fit(np.array(sentenceSet))
    u = pca.components_[0]
    u = np.multiply(u, np.transpose(u))

    # occurs if we have less sentences than embeddings_size
    if len(u) < embeddingSize:
        for i in range(embeddingSize - len(u)):
            u = np.append(u, [0])

            # remove the projections of the average vectors on their first principal component
    # (“common component removal”).
    sentenceVectors = []
    for sentenceVector in sentenceSet:
        sentenceVectors.append(np.subtract(sentenceVector, np.multiply(u, sentenceVector)))
    return sentenceVectors


# 获取训练数据
def gettrainData():
    question_path = r'./shuxueTest/shuxueTrainData.pkl'
    longtextdata1 = pickle.load(open(question_path, 'rb'))
    longtextdata1 = longtextdata1['question_text']
    traind = longtextdata1[:5000]
    traindata = list(map(lambda x: x.split(' '), traind))
    return traindata


def saveIndex(sentence_vecs):
    corpus_len = len(sentence_vecs)
    print(corpus_len)
    index = np.empty(shape=(corpus_len, 200), dtype=np.float32)
    for docno, vector in enumerate(sentence_vecs):
        if isinstance(vector, np.ndarray):
            pass
        elif scipy.sparse.issparse(vector):
            vector = vector.toarray().flatten()
        else:
            vector = matutils.unitvec(matutils.sparse2full(vector, 200))
        index[docno] = vector
    return index


# 计算矩阵与向量余弦相识度
def cosine_Matrix(_matrixA, vecB):
    _matrixA_matrixB = np.dot(_matrixA, vecB.T).T
    _matrixA_norm = np.sqrt(np.multiply(_matrixA, _matrixA).sum(axis=1))
    vecB_norm = np.linalg.norm(vecB)
    return np.divide(_matrixA_matrixB, _matrixA_norm * vecB_norm.transpose())


def trainWordVectAvg(input_pca_sentence,model):
   # traindata = gettrainData()
    #r=open(input_pca_sentence,'r',encoding='utf-8')
    traindata=input_pca_sentence


    #traindata = r.readlines()
    #traindata=[lines.split(' ') for lines in traindata]

    dictionary = corpora.Dictionary(traindata)  ##得到词典

    token2id = dictionary.token2id
    charLen = dictionary.num_pos
    corpus = [dictionary.doc2bow(text) for text in traindata]  ##统计每篇文章中每个词出现的次数:[(词编号id,次数number)]
    print('dictionary prepared!')
    tfidf = models.TfidfModel(corpus=corpus, dictionary=dictionary)
    wdfs = tfidf.dfs
    #corpus_tfidf = tfidf[corpus]
    #model = Word2Vec(traindata, size=200, window=5, min_count=1, workers=4)
    #model = Word2Vec.load('C:\linux_web_download\\test.model')

    # 词向量求平均得到句向量
    #sentence_vecs = sentenceByWordVectAvg(traindata, model, 200)
    # 词向量tfidf加权得到句向量
    #sentence_vecs = sentenceByW2VTfidf(corpus_tfidf, token2id, traindata, model, 200)
    # sentence2vec：词向量加权-PCA
    Sentence_list = []
    for td in traindata:
        vecs = []
        for s in td:
            try:
                w = Word(s, model[s])
            except KeyError:
                w=Word(s,np.zeros([200]))
            vecs.append(w)
        sentence = Sentence(vecs)
        Sentence_list.append(sentence)
    sentence_vecs = sentence2vec(wdfs, token2id, Sentence_list, 200, charLen)

    query = sentence_vecs[0]
    #print(query)
    index = saveIndex(sentence_vecs)
    query = sentence_vecs[0]
    # 计算相似度
    cosresult = cosine_Matrix(index, query)
    cosresult = cosresult.tolist()
    sort_cosresult = sorted(cosresult, reverse=True)

    #print(sort_cosresult)
    #print(sentence_vecs)
    list_result = []
    for i in sort_cosresult:
        idx = cosresult.index(i)
        #print(i, '===', traindata[idx])
        list_result.append([i,''.join(traindata[idx])])

    return list_result
    #print(traindata[0])

#trainWordVectAvg('C:\新建文件夹\新建文件夹 (4)\学习资料\project-1\\news_file_cut_word.txt')