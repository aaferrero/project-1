#coding=gbk
import pymysql
import sys
import json
from json import *
# 连接database
import numpy as np
import os
#sub=['报道', '微软', 'Windows', '更好地']
#veb=['称', '拿到', '理解', '理解']
#veb=['见到','高大']
#sub=['南', '记者']
#veb_symbol=[[2, 'SBV',4], [6, 'SBV',5], [18, 'SBV',6], [18, 'SBV',7]]
def concat_verb_sub(list_verb,list_sub,list_verb_symbol):
    zipped = zip(list_sub, list_verb)
    list_veb_sub = [list(element) for element in zipped]
    zipped_verb_sub_symbol = zip(list_veb_sub, list_verb_symbol)

    list_zipped_verb_sub_symbol = [list(element) for element in zipped_verb_sub_symbol]
    #print(list_zipped_verb_sub_symbol)
    #start_index = 0
    del_list = []
    #while start_index<len(list_zipped_verb_sub_symbol):
    #for start_index1 in range(start_index, len(list_zipped_verb_sub_symbol)):
    for number in range(1, len(list_zipped_verb_sub_symbol)):

        if list_zipped_verb_sub_symbol[number][1][0] == list_zipped_verb_sub_symbol[number - 1][1][0] and abs(
                        list_zipped_verb_sub_symbol[number][1][2] - list_zipped_verb_sub_symbol[number - 1][1][2]) == 1:
            list_zipped_verb_sub_symbol[number][0][0] = list_zipped_verb_sub_symbol[number - 1][0][0] \
                                                        + list_zipped_verb_sub_symbol[number][0][0]
            del_list.append(list_zipped_verb_sub_symbol[number - 1])
        elif list_zipped_verb_sub_symbol[number][1][0] == list_zipped_verb_sub_symbol[number - 1][1][0] and abs(
                        list_zipped_verb_sub_symbol[number][1][2] - list_zipped_verb_sub_symbol[number - 1][1][2]) != 1:
            del_list.append(list_zipped_verb_sub_symbol[number - 1])
            del_list.append(list_zipped_verb_sub_symbol[number])
    for del_element in del_list:
        if del_element in list_zipped_verb_sub_symbol:
            list_zipped_verb_sub_symbol.remove(del_element)
    return list_zipped_verb_sub_symbol

#print(concat_verb_sub(veb,sub,veb_symbol))


#print([element for element in list_zipped_verb_sub_symbol])

'''
from cnocr import CnOcr

import mxnet
ocr = CnOcr()
img_fp = 'C:\ltp_data_v3.4.0\\test3.png'
img = mxnet.image.imread(img_fp, 1)
res = ocr.ocr(img)
print("Predicted Chars:", res)
sys.exit()



conn = pymysql.connect(
    host='rm-8vbwj6507z6465505ro.mysql.zhangbei.rds.aliyuncs.com',
user ='root', password ='AI@2019@ai',
database ='stu_db',
charset ='utf8')

cursor = conn.cursor()
sql = "SELECT feature,content FROM news_chinese where author='王子江 张寒'"
cursor.execute(sql)
res = cursor.fetchone() #第一次执行
print(res[0])

print(res[1].replace('\\n','\n'))
sys.exit()




import os
from pyltp import Segmentor
from pyltp import SentenceSplitter
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
import pandas as pd

files=pd.read_csv('news_chinese.csv')
#print(files.iloc[-1])
for number in range(1):
    print('------------------------------------------------')
    print('content:\n',files.iloc[number]['content'].replace('\\n', '\n'))
    sentence2=files.iloc[number]['content'].replace('\\n', '\n')
    sentence2='但韩国网友对“韩国海军陆战队世界第二”的说法不以为然'
    cws_model_path = 'C:\ltp_data_v3.4.0\cws.model'
    segmentor = Segmentor()
    segmentor.load(cws_model_path)
    words = segmentor.segment(sentence2)

    sentence1 = '\t'.join(words)
    m=0
    list_word=[]
    for word in sentence1:
        list_word.append(word+'\n\n')
    print(sentence1.split('\t'))
    list_sentence = sentence1.split('\t')
    print('\t'.join(words))
    segmentor.release()

    from pyltp import Postagger

    postagger = Postagger()  # 初始化实例
    postagger.load('C:\ltp_data_v3.4.0\pos.model')  # 加载模型
    list_sentence=['但', '韩国', '网友', '对', '“', '韩国', '海军', '陆战队', '世界', '第二', '”', '的', '说法','不以为然']
    # 分词结果
    postags = postagger.postag(list_sentence)  # 词性标注

    sentence_postags = '\t'.join(postags)
    print('\t'.join(postags))
    list_postags = sentence_postags.split('\t')
    postagger.release()  # 释放模型

    from pyltp import Parser

    par_model_path='C:\ltp_data_v3.4.0\parser.model'
    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型
    arcs = parser.parse(list_sentence, list_postags)
    print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
    parser.release()  # 释放模型

    recognizer = NamedEntityRecognizer()  # 初始化实例

    # LTP_DATA_DIR='C:\新建文件夹\新建文件夹 (4)\学习资料\project-1\ltp_data_v3.4.0'

    ner_model_path = 'C:\ltp_data_v3.4.0\\ner.model'
    recognizer.load(ner_model_path)  # 加载模型
    segmentor = Segmentor()
    netags = recognizer.recognize(list_sentence, list_postags)  # 命名实体识别

    print('\t'.join(netags))
    list_netags = '\t'.join(netags).split('\t')
    dict_netags = {}
    n = 0
    for netag in list_netags:
        if list_sentence[n]!='\r\n':
            print(list_sentence[n]+'-'+netag+'|',end='')
        else:
            print(list_sentence[n])
        #if netag != 'O':
            #dict_netags[list_sentence[n] + str(n) + list_postags[n]] = netag
        n += 1

    print(dict_netags)

    recognizer.release()  # 释放模型
    segmentor.release()

sys.exit()





import tensorflow as tf
import numpy as np
import sys
t1 =([[[1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]]],[[[1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]]])
t2 = tf.constant([[7, 8, 9], [10, 11, 12]])

concat0=tf.concat(t1, 2)
s=tf.Session()

print(s.run(concat0))

sys.exit()


batch_size = 3
input_dim = 2
output_dim = 4

inputs = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_dim))
previous_state = (tf.random_normal(shape=(batch_size, output_dim)), tf.random_normal(shape=(batch_size, output_dim)))

cell = tf.contrib.rnn.BasicLSTMCell(num_units=output_dim)
output, (state_c, state_h) = cell(inputs, previous_state)

X = np.ones(shape=(batch_size, input_dim))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    o, s_c, s_h = sess.run([output, state_c, state_h], feed_dict={inputs: X})

    print(X)
    print(previous_state[0].eval())
    print('---------------------------------')
    print(previous_state[1].eval())
    print('---------------------------------')
    print(o)
    print('---------------------------------')
    print(s_c)
    print('---------------------------------')
    print(s_h)
    print('---------------------------------')
'''