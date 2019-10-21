#coding=gbk
import pandas as pd
import jieba
import json
from collections import defaultdict
from functools import lru_cache
from gensim.models import Word2Vec,word2vec
from pyltp import Parser
#from processfile import concat_verb_sub
import numpy as np
import sys
from pyltp import Postagger
import re

from zhon.hanzi import punctuation
from pca_sentence_vector import trainWordVectAvg



#list_sentence=['��', '����', '����', '��', '��', '����', '����', '½ս��', '����', '�ڶ�', '��', '��', '˵��','����ΪȻ']
    # �ִʽ��

new_punctuation = punctuation + ','


@lru_cache(maxsize=2**10)
def get_related_words(initial_words, model):
    """
    @initial_words are initial words we already know
    @model is the word2vec model
    """

    unseen = initial_words

    seen = defaultdict(int)

    max_size = 2800  # could be greater

    while unseen and len(seen) < max_size:
        if len(seen) % 50 == 0:
            print('seen length : {}'.format(len(seen)))

        node = unseen[0]
        unseen=unseen[1:]

        new_expanding = [w for w, s in model.most_similar(node, topn=20)]


        unseen += tuple(new_expanding)

        seen[node] += get_related_scores(node,initial_words,model)

        # optimal: 1. score function could be revised
        # optimal: 2. using dymanic programming to reduce computing time

    return seen

def get_related_scores(node,words,model):
    np_socres = np.zeros([20 * len(words)])
    number=0
    for word in words:
        expanding_words = [w for w, s in model.most_similar(word, topn=20)]
        for expanding_word in expanding_words:
            np_socres[number]=model.similarity(node,expanding_word)
    number+=1
    return np.mean(np_socres)
'''
model = Word2Vec.load('C:\linux_web_download\\test.model')
related_words = get_related_words(('˵', '��ʾ'), model)
c=list(related_words.items())
c1=sorted(c, key=lambda x: x[1], reverse=True)
print(c1)
with open('said_txt_latest.json', 'w+') as f:

        jsonArr = json.dumps(c1, ensure_ascii=False)

        f.write(jsonArr)
'''
#model = Word2Vec.load('C:\linux_web_download\\test.model')
#files=pd.read_csv('news_chinese.csv')

def get_one_content(files,number):
    sentence = files.iloc[number]['content']
    print('------------------------------------------------')
    print(sentence)
    try:
        sentence = sentence.replace('\u3000', '')
        sentence = sentence.replace('\r', '')
        para_split_sentence = sentence.split('\n')[:-1]
        #print(para_split_sentence)
    except AttributeError:
        return None
    else:
        return para_split_sentence

    #print(sentence)
    #print('.........................................')
    #for one_sentence in sentence_split:
        #print(one_sentence)
    #print('----------------------------------------------')


def get_related_sub_verb_in_onetxt(files,number,list_tells,postagger,parser):
    par_model_path = 'C:\ltp_data_v3.4.0\parser.model'
    if get_one_content(files, number)==None:
        return None,None
    get_para_split_sentence = ['\t'.join(jieba.cut(sentence)).split('\t') for sentence in
                               get_one_content(files, number)]
    #get_para_split_sentence = [['����', 'Ŀǰ', '����', '��', '����', '��', 'һ��', '�ֻ�', '5', 'ӵ��', '5.5', '��', '1080P', '����', 'AMOLED', '��ʾ��', '��', '6G', '/', '8GB', ' ', 'RAM', '��', '64GB', '/', '128GB', ' ', 'ROM', '��', '˫', '1600', '��', '����ͷ', '��', '����', '��', '��', '��ϲ', '��', '��']]

    list_sentence = []
    sub_veb_txt_dict = {}
    for sentence_split in get_para_split_sentence:
        #postagger = Postagger()  # ��ʼ��ʵ��
        #postagger.load('C:\ltp_data_v3.4.0\pos.model')  # ����ģ��
        postags = postagger.postag(sentence_split)  # ���Ա�ע
        sentence_postags = '\t'.join(postags)
        #print(list(zip(list(postags),sentence_split)))
        list_postags = sentence_postags.split('\t')
        #postagger.release()  # �ͷ�ģ��
        #parser = Parser()  # ��ʼ��ʵ��
        #parser.load(par_model_path)  # ����ģ��
        arcs = parser.parse(sentence_split, list_postags)
        list_all_arcs = [[arc.head, arc.relation,index] for index,arc in enumerate(arcs)]
        list_arcs_sentence_split = list(zip(list_all_arcs,sentence_split))
        print(list_arcs_sentence_split)

        list_arcs = [[arc.head, arc.relation,index] for index,arc in enumerate(arcs) if arc.relation == 'SBV']
        start_index = 0
        list_sub_words = []
        list_all_vebs = []
        c = 0
        for sub_word in list_arcs:
            index_sub = list_all_arcs.index(sub_word, start_index)
            start_index = index_sub + 1
            list_sub_words.append(sentence_split[index_sub])
        #print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
        #print('ÿ�����������', list_sub_words)
        #print(list_arcs)
        #parser.release()  # �ͷ�ģ��
        for arc_word in list_arcs:
            list_all_vebs.append(sentence_split[int(arc_word[0]) - 1])
        #print('ÿ��������ν�', list_all_vebs)

        if list_arcs != []:
            list_concat_verb_sub = concat_verb_sub(
                list_all_vebs,list_sub_words,list_arcs)
            #print(list_sub_words,list_all_vebs)
            #print(list_concat_verb_sub)
            list_arc_word=[]
            for arc_word in list_concat_verb_sub:
                if len(list_arcs_sentence_split) <= arc_word[1][0]:
                    list_arc_word.append(arc_word)
                    continue
                if '��' in arc_word[0][0]:
                    continue
                if arc_word[0][1] in list_tells or (list_arcs_sentence_split[arc_word[1][0]][0][0] == arc_word[1][0] and
                                                            list_arcs_sentence_split[arc_word[1][0]][1] in list_tells):
                    #print('������ν��Ϊ������', list_arcs, arc_word, sentence_split[int(arc_word[0]) - 1])
                    #print('������ÿ�䣺', ''.join(sentence_split))
                    #print(arc_word[0])

                    c = 1
                    if list_arcs_sentence_split[arc_word[1][0]][0][0] == arc_word[1][0] and \
                                    list_arcs_sentence_split[arc_word[1][0]][1] not in new_punctuation:
                        arc_word.append(1)
                        list_arc_word.append(arc_word)
                    else:
                        list_arc_word.append(arc_word)

            sub_veb_txt_dict[''.join(sentence_split)] = [sentence_split,
                list_sub_words, list_all_vebs,list_arc_word]
            list_sentence.append(''.join(sentence_split))

    return sub_veb_txt_dict , list_sentence


def match_txt(sentence,sub_veb_txt_dict):
    list_position = []
    list_match_txt = []
    if len(sub_veb_txt_dict[sentence][3])>=0:
        for veb_number in range(len(sub_veb_txt_dict[sentence][3])):

            if veb_number == len(sub_veb_txt_dict[sentence][3])-1:
                match_sentence = ''.join(
                    sub_veb_txt_dict[sentence][0][sub_veb_txt_dict[sentence][3][veb_number][1][0]:])

            else:
                match_sentence = ''.join(
                    sub_veb_txt_dict[sentence][0][sub_veb_txt_dict[sentence][3][veb_number][1][0]:sub_veb_txt_dict[sentence][3][veb_number+1][1][0]])
            #print('match_sentence:',match_sentence)
            if len(sub_veb_txt_dict[sentence][3][veb_number])==3:

                if match_first_symbol(match_sentence) =='':
                    if len(list_match_txt)!=0:
                        #match_sentence = ''.join(
                            #sub_veb_txt_dict[sentence][0][sub_veb_txt_dict[sentence][3][veb_number-1][1][0]:])
                        list_match_txt[-1][1]= list_match_txt[-1][1] + match_sentence

                    break
                else:
                    list_match_txt.append([sub_veb_txt_dict[sentence][3][veb_number][0],match_first_symbol(match_sentence)])
                #print('list_match_txt', list_match_txt)
            else:
                list_match_txt.append(
                    [sub_veb_txt_dict[sentence][3][veb_number][0], match_sentence])


    return list_match_txt

def match_first_symbol(string):
    #new_punctuation = punctuation + ','               #ƥ���һ�������ſ�ʼ�����
    #print('match_first_symbol(string):',string)
    string_pattern2 = r"[%s+]" % new_punctuation
    pattern = re.compile(string_pattern2)
    result = pattern.findall(string)
    if '.' in result:
        result.remove('.')
    if len(result) > 0:
        #print('match_first_symbol(string):', string[string.find('{s}'.format(s=result[0]), 0, len(string)):])
        return string[string.find('{s}'.format(s=result[0]), 0, len(string)):]

    else:
        return ''



def regulation_match(list_match_txt):
    string_pattern1 = r'.*?(��.*?��)'
    #print(punctuation)
    string_pattern2 = r"[%s+]" %punctuation

    list_result = []
    for match_part in list_match_txt:
        string=match_part[1]
        pattern = re.compile(string_pattern1)
        result = pattern.findall(string)

        if False:
            if len(result) > 1:
                result = [','.join(result)]
                #print(result)
            else:
                result = [result[0]]
                #print(result)
            #dict_result['symbol_q'] = result
            list_result.append([match_part[0],result])
        else:
            pattern = re.compile(string_pattern2)
            result = pattern.findall(string)

            if len(result)!=0:
                start = string.find('{s}'.format(s=result[0]), 0, len(string))
                #start1 = string.find('{s}'.format(s=result[1]), start + 1, len(string))
                end = string.rfind('{s}'.format(s=result[-1]), 0, len(string))
                #print('string,end',string,end)
                split_symbol_list = ['��','��','��','?']
                letter_index_  = 0
                letter_indexs = []
                string_split_list =[]
                for letter_index,letter in enumerate(string):
                    if letter in split_symbol_list:
                        #if letter_index ==len(string)-1:
                            #break
                        letter_indexs.append(letter_index)
                        if letter_index_ ==0:
                            if start != 0:
                                string_split_list.append(
                                    string[:letter_indexs[letter_index_]])
                            else:
                                first_string_split = string[start + 1:letter_indexs[letter_index_]]
                                if len(first_string_split)>0:
                                    string_split_list.append(first_string_split)
                                else:
                                    print('error:',string)

                        else:
                            string_split = string[letter_indexs[letter_index_ - 1]+1:letter_indexs[letter_index_]]
                            if len(string_split)>0:
                                string_split_list.append(string_split)

                        letter_index_ += 1
                if string_split_list!=[]:
                    list_result.append([match_part[0], string_split_list])
                else:
                    list_result.append([match_part[0], [string[start + 1 if start ==0 else 0:end]]])
            else:
                list_result.append([match_part[0], [string]])
            #print('string_split_list',string_split_list)
            #print('string_split_list', string)


    return list_result


def compare_txt_similar(list_result,ratio,model):
    results=[]
    #print('list_result:',list_result)
    for list_result_ in list_result:
        input_pca_sentence = [list(jieba.cut(sentence)) for sentence in list_result_[1]]
        print('input_pca_sentence',input_pca_sentence)

        if len(input_pca_sentence)>1:
            #print(list_result_[0])
            pca_result = trainWordVectAvg(input_pca_sentence,model)
            print('pca_result:',pca_result)
            pca_result = [sentence_[1] for sentence_ in pca_result if sentence_[0]>ratio]
            results.append([list_result_[0],''.join(pca_result)])
        elif len(input_pca_sentence) == 0 or len(input_pca_sentence[0])==0:
            print('��ƥ��')
            return []
        else:
            results.append([list_result_[0], input_pca_sentence[0]])
    return results


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




'''
dict1={'�ֽ�˹�ȴ�41������������ף��������������߳��������³ǵ�Τ������������˵���������λã������������ţ����Ҹо��ܺã������ϵ�����һ�죬���벻��Ҫ��41����ô������': [['�ֽ�˹', '�ȴ�', '41', '��', '��', '����', '������', '��', '��', '��', '����', '����', '��', '��', '�߳�', '������', '�³�', '��', 'Τ����', '����', '��', '��', '˵', '��', '��', '����', '�λ�', '��', '����', '��������', '��', '��', '��', '�о�', '��', '��', '��', '��', '��', '�ϵ�', '��', '��', 'һ��', '��', '��', '�벻��', 'Ҫ', '��', '41', '��', '��ô', '��', '��', '��'], ['�ֽ�˹', '��', '�λ�', '��', '��'], ['�ȴ�', '˵', '����', '�о�', '�ϵ�'], [[['��', '˵'], [23, 'SBV', 21]], [['��', '�о�'], [34, 'SBV', 32]]]], '��Ϥ��61����ֽ�˹������1975������25��Ů����ޱ�أ�������ǹɱ������20����ֽ�˹�����������������һ��ıɱ�ﶨ�����������Ĺؼ��������ֳ���һ��ͷ����': [['��Ϥ', '��', '61', '��', '��', '�ֽ�˹', '����', '��', '1975', '��', '����', '25', '��', 'Ů��', '��ޱ��', '��', '��', '��', '��', 'ǹɱ', '��', '����', '20', '��', '��', '�ֽ�˹', '��', '����', '����', '��', '���', '��', '��', 'һ��', 'ıɱ��', '����', '��', '��', '��', '��', '����', '��', '�ؼ�', '��', '��', '��', '�ֳ�', '��', 'һ��', 'ͷ��', '��'], ['�ֽ�˹', '�ؼ�'], ['����', '��'], []], '�������������Ľ��������߸��˹۵㣬�뻷�����޹ء���ԭ�����Լ����г������ֺ�����δ����վ֤ʵ���Ա����Լ�����ȫ�����߲������ݡ����ֵ���ʵ�ԡ������ԡ���ʱ�Ա�վ�����κα�֤���ŵ������߽����ο����������к�ʵ������ݡ�': [['����', '����', '��', '����', '��', '����', '����', '���˹۵�', '��', '��', '������', '�޹�', '��', '��', 'ԭ����', '�Լ�', '����', '����', '����', '��', '����', 'δ��', '��վ', '֤ʵ', '��', '��', '����', '�Լ�', '����', 'ȫ��', '����', '����', '����', '��', '����', '��', '��ʵ��', '��', '������', '��', '��ʱ��', '��վ', '��', '��', '�κ�', '��֤', '��', '��ŵ', '��', '��', '����', '����', '�ο�', '��', '��', '��', '����', '��ʵ', '���', '����', '��'], ['����', 'ԭ����', '��վ', '����', '����', '��վ'], ['����', '֤ʵ', '֤ʵ', '��', '��', '��'], [[['ԭ����', '֤ʵ'], [24, 'SBV', 14]], [['��վ', '֤ʵ'], [24, 'SBV', 22]]]], '���ר���ܼ�����������˵������ͷ�����ﲻ�ǽ����ڿ�ѧ�Ļ����ϡ�����֯����1��Ҫ��Ժ�����ֽ�˹�Ķ��': [['���', 'ר��', '�ܼ�', '����', '��', '����˵', '��', '����', 'ͷ��', '����', '����', '����', '��', '��ѧ', '��', '����', '��', '��', '��', '��֯', '����', '1', '��', 'Ҫ��', '��Ժ', '����', '�ֽ�˹', '��', '����', '��'], ['�ܼ�', '����', '��֯', '��Ժ'], ['����˵', '����', 'Ҫ��', '����'], [[['��֯', 'Ҫ��'], [24, 'SBV', 19]]]], '������6��19�յ� ����ý����������������һ������1976����Ϊһ��ͷ������ıɱ����41���������Э���³ɹ�ϴˢ�������ڱ���15�ջ��͡�': [['������', '6', '��', '19', '�յ�', ' ', '��', '��ý', '����', '��', '����', '������', 'һ��', '����', '1976', '��', '��Ϊ', 'һ��', 'ͷ��', '����', 'ıɱ��', '��', '���', '41', '��', '��', '��', '����', '��', 'Э��', '��', '�ɹ�', 'ϴˢ', '����', '��', '��', '����', '15', '��', '����', '��'], ['��ý', '����', 'ͷ��'], ['����', 'ϴˢ', '����'], []], '����������ѧ������ѧԺ�ġ������Ŀ��Ϊ��Э����������ԩ��֮��������ˣ����������ֽ�˹�����ֵ��꾯��ʵ���ҵķ���ʦ�������ֳ��ҵ���һ��ͷ�������϶��ð����ֽ�˹�йء�': [['��', '������', '��ѧ', '����', '��ѧԺ', '��', '��', '���', '��Ŀ', '��', 'Ϊ��', 'Э��', '��', '��', '����', 'ԩ��', '֮��', '��', '������', '��', '����', '����', '�ֽ�˹', '��', '����', '����', '����', 'ʵ����', '��', '����ʦ', '����', '��', '�ֳ�', '�ҵ�', '��', 'һ��', 'ͷ��', '��', '��', '�϶�', '�ð�', '��', '�ֽ�˹', '�й�', '��'], ['��Ŀ', '����ʦ', '�ð�'], ['����', '�϶�', '�й�'], [[['��Ŀ', '����'], [22, 'SBV', 8]], [['����ʦ', '�϶�'], [40, 'SBV', 29]]]]}


print(dict1)
#regulation_match(1)

sentence1='�ֽ�˹�ȴ�41������������ף��������������߳��������³ǵ�Τ����������' \
          '��˵���������λã������������ţ����Ҹо��ܺã������ϵ�����һ�죬���벻��Ҫ��41����ô������'
#cut_sentence=jieba.cut(sentence1)
#print(list(cut_sentence))
#print(match_txt(sentence1,dict1))

match_txt_  = match_txt(sentence1,dict1)
print('match_txt',match_txt_)
print(regulation_match(match_txt_))
regulation_match = regulation_match(match_txt_)

print('regulation_match',regulation_match)

compare_txt_similar(regulation_match,0.5)

'''






