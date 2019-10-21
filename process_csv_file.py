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



#list_sentence=['但', '韩国', '网友', '对', '“', '韩国', '海军', '陆战队', '世界', '第二', '”', '的', '说法','不以为然']
    # 分词结果

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
related_words = get_related_words(('说', '表示'), model)
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
    #get_para_split_sentence = [['按照', '目前', '掌握', '的', '资料', '，', '一加', '手机', '5', '拥有', '5.5', '寸', '1080P', '三星', 'AMOLED', '显示屏', '、', '6G', '/', '8GB', ' ', 'RAM', '，', '64GB', '/', '128GB', ' ', 'ROM', '，', '双', '1600', '万', '摄像头', '，', '备货', '量', '“', '惊喜', '”', '。']]

    list_sentence = []
    sub_veb_txt_dict = {}
    for sentence_split in get_para_split_sentence:
        #postagger = Postagger()  # 初始化实例
        #postagger.load('C:\ltp_data_v3.4.0\pos.model')  # 加载模型
        postags = postagger.postag(sentence_split)  # 词性标注
        sentence_postags = '\t'.join(postags)
        #print(list(zip(list(postags),sentence_split)))
        list_postags = sentence_postags.split('\t')
        #postagger.release()  # 释放模型
        #parser = Parser()  # 初始化实例
        #parser.load(par_model_path)  # 加载模型
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
        #print('每句中所有主语：', list_sub_words)
        #print(list_arcs)
        #parser.release()  # 释放模型
        for arc_word in list_arcs:
            list_all_vebs.append(sentence_split[int(arc_word[0]) - 1])
        #print('每句中所有谓语：', list_all_vebs)

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
                if '我' in arc_word[0][0]:
                    continue
                if arc_word[0][1] in list_tells or (list_arcs_sentence_split[arc_word[1][0]][0][0] == arc_word[1][0] and
                                                            list_arcs_sentence_split[arc_word[1][0]][1] in list_tells):
                    #print('段落中谓语为称述；', list_arcs, arc_word, sentence_split[int(arc_word[0]) - 1])
                    #print('段落中每句：', ''.join(sentence_split))
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
    #new_punctuation = punctuation + ','               #匹配第一个标点符号开始的语句
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
    string_pattern1 = r'.*?(“.*?”)'
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
                split_symbol_list = ['。','”','？','?']
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
            print('无匹配')
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
dict1={'沃金斯等待41年后终于真相大白，被获撤销罪名后，走出底特律下城的韦恩郡监狱。他说：“这真梦幻，令人难以置信，但我感觉很好，我早料到有这一天，但想不到要等41年那么长。”': [['沃金斯', '等待', '41', '年', '后', '终于', '真相大白', '，', '被', '获', '撤销', '罪名', '后', '，', '走出', '底特律', '下城', '的', '韦恩郡', '监狱', '。', '他', '说', '：', '“', '这真', '梦幻', '，', '令人', '难以置信', '，', '但', '我', '感觉', '很', '好', '，', '我', '早', '料到', '有', '这', '一天', '，', '但', '想不到', '要', '等', '41', '年', '那么', '长', '。', '”'], ['沃金斯', '他', '梦幻', '我', '我'], ['等待', '说', '令人', '感觉', '料到'], [[['他', '说'], [23, 'SBV', 21]], [['我', '感觉'], [34, 'SBV', 32]]]], '据悉，61岁的沃金斯被控在1975年抢劫25岁女子伊薇特，并将其枪杀。当年20岁的沃金斯被警方逮捕，最后被以一级谋杀罪定罪，而他被定罪的关键仅是在现场的一根头发。': [['据悉', '，', '61', '岁', '的', '沃金斯', '被控', '在', '1975', '年', '抢劫', '25', '岁', '女子', '伊薇特', '，', '并', '将', '其', '枪杀', '。', '当年', '20', '岁', '的', '沃金斯', '被', '警方', '逮捕', '，', '最后', '被', '以', '一级', '谋杀罪', '定罪', '，', '而', '他', '被', '定罪', '的', '关键', '仅', '是', '在', '现场', '的', '一根', '头发', '。'], ['沃金斯', '关键'], ['被控', '是'], []], '免责声明：本文仅代表作者个人观点，与环球网无关。其原创性以及文中陈述文字和内容未经本站证实，对本文以及其中全部或者部分内容、文字的真实性、完整性、及时性本站不作任何保证或承诺，请读者仅作参考，并请自行核实相关内容。': [['免责', '声明', '：', '本文', '仅', '代表', '作者', '个人观点', '，', '与', '环球网', '无关', '。', '其', '原创性', '以及', '文中', '陈述', '文字', '和', '内容', '未经', '本站', '证实', '，', '对', '本文', '以及', '其中', '全部', '或者', '部分', '内容', '、', '文字', '的', '真实性', '、', '完整性', '、', '及时性', '本站', '不', '作', '任何', '保证', '或', '承诺', '，', '请', '读者', '仅作', '参考', '，', '并', '请', '自行', '核实', '相关', '内容', '。'], ['本文', '原创性', '本站', '本文', '内容', '本站'], ['代表', '证实', '证实', '作', '作', '作'], [[['原创性', '证实'], [24, 'SBV', 14]], [['本站', '证实'], [24, 'SBV', 22]]]], '清白专案总监马拉米契尔说，根据头发定罪不是建立在科学的基础上。该组织今年1月要求法院撤销沃金斯的定罪。': [['清白', '专案', '总监', '马拉', '米', '契尔说', '，', '根据', '头发', '定罪', '不是', '建立', '在', '科学', '的', '基础', '上', '。', '该', '组织', '今年', '1', '月', '要求', '法院', '撤销', '沃金斯', '的', '定罪', '。'], ['总监', '定罪', '组织', '法院'], ['契尔说', '不是', '要求', '撤销'], [[['组织', '要求'], [24, 'SBV', 19]]]], '中新网6月19日电 据外媒报道，美国底特律一名男子1976年因为一根头发被定谋杀罪，监禁41年后，终于在协助下成功洗刷罪名，于本月15日获释。': [['中新网', '6', '月', '19', '日电', ' ', '据', '外媒', '报道', '，', '美国', '底特律', '一名', '男子', '1976', '年', '因为', '一根', '头发', '被定', '谋杀罪', '，', '监禁', '41', '年', '后', '，', '终于', '在', '协助', '下', '成功', '洗刷', '罪名', '，', '于', '本月', '15', '日', '获释', '。'], ['外媒', '男子', '头发'], ['报道', '洗刷', '被定'], []], '西密西根大学库利法学院的“清白项目”为了协助更多蒙受冤狱之苦的受刑人，积极帮助沃金斯，发现当年警局实验室的分析师基于在现场找到的一根头发，而认定该案与沃金斯有关。': [['西', '密西根', '大学', '库利', '法学院', '的', '“', '清白', '项目', '”', '为了', '协助', '更', '多', '蒙受', '冤狱', '之苦', '的', '受刑人', '，', '积极', '帮助', '沃金斯', '，', '发现', '当年', '警局', '实验室', '的', '分析师', '基于', '在', '现场', '找到', '的', '一根', '头发', '，', '而', '认定', '该案', '与', '沃金斯', '有关', '。'], ['项目', '分析师', '该案'], ['帮助', '认定', '有关'], [[['项目', '帮助'], [22, 'SBV', 8]], [['分析师', '认定'], [40, 'SBV', 29]]]]}


print(dict1)
#regulation_match(1)

sentence1='沃金斯等待41年后终于真相大白，被获撤销罪名后，走出底特律下城的韦恩郡监狱。' \
          '他说：“这真梦幻，令人难以置信，但我感觉很好，我早料到有这一天，但想不到要等41年那么长。”'
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






