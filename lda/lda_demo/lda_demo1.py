import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


"""
LDA分析步骤：
读取数据，并分词
去除停用词
构建Tfidf矩阵，每一行代表一个test的文档，每一列代表一个词语的tfidf值
LDA分析（fit和transform），输出结果
"""

# 1.读取数据与分词
filepaths = ['../data/test1.txt', '../data/test2.txt', '../data/test3.txt', '../data/test4.txt']
docs = [open(f, mode="r", encoding="utf-8").read() for f in filepaths]
docs = [jieba.lcut(doc) for doc in docs]
print(docs)
print(len(docs))

"""
2.去除停止词
实际操作中需要自己构建停用词表，然后剔除掉这些无意义的词语，在本文中去停止词操作比较简单粗暴了点，只保留词语长度大于1的。
"""
docs = [[w
         for w in doc
         if len(w) > 1]
         for doc in docs]
# sklearn默认分析的语言是英文，我们要组织成类似英文那样以空格间隔的语言形式。
# corpus现在是一个列表，列表中有四个字符串。
# 每个字符串就是一个文档
corpus = [' '.join(doc) for doc in docs]
print(corpus)


"""
3. 构建Tfidf矩阵
每一行代表一个test的文档，每一列代表一个词语的tfidf值。学过的sklearn的都知道fit和transform的意义，
如果对tfidf不懂的可以查看咱们之前分享的文章。
"""
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(corpus)
print(tfidf_matrix)
"""
tfidf_matrix是 4x496，即4行496列，其中
4行指的是四个文档
496列是496个词语（也就是语料中一共出现了496个词语）
"""

"""
4. LDA分析（fit和transform）
同上，这里也分为fit和transform，由于我们有预先的知识，知道这四个文档来源于三国和三体，所以话题数K天然的等于2，即 n_topics=2 。
由于LDA属于聚类分析的一种，而聚类分析过程中会随机初始化，为了保证你也能得到与大邓一样的运行结果，我设置了random_state=123456。
当然设置成别的数字也可以，这里的random_state相当于口令，咱们两个口令一致才能得到相同的答案。如果你换了random_state，
那么咱们两个得到的结果可能会有出入。
"""
lda = LatentDirichletAllocation(n_components=2, random_state=123456)
docres = lda.fit_transform(tfidf_matrix)
print(docres)
"""
得到的结果是 4*2 的矩阵。行表示文档，列表示话题。我们将第一列认定为话题1，第二列认定为话题2
[[0.9179533  0.0820467 ] 属于话题1
 [0.91490373 0.08509627] 属于话题1
 [0.08152025 0.91847975] 属于话题2
 [0.91247862 0.08752138]] 属于话题2
"""

