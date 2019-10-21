import pandas as pd
import jieba
import json
from gensim.models import Word2Vec,word2vec
from flask import Flask, render_template,send_file
from flask import request
from flask import jsonify
from process_csv_file import get_one_content
from process_csv_file import get_related_sub_verb_in_onetxt , regulation_match , compare_txt_similar , match_txt
from pyltp import Parser
from pyltp import Postagger

def news_file_cut_word():
    files = pd.read_csv('news_chinese.csv')
    f=open('news_file_cut_word.txt','w+',encoding='utf-8')
    for number in range(len(files)):
        if get_one_content(files,number)!=None:
            for sentence in get_one_content(files, number):
                if sentence != None:
                    seg_list = jieba.cut(sentence)
                    seg_lines = ' '.join(seg_list)+'\n'
                    f.write(seg_lines)

app = Flask(__name__, static_url_path='')
@app.route('/',methods=['POST','GET'])
def start():
    file_st = 0
    file_end = 5
    text = request.args.get("content_number")
    print('content_number',text.split('-'))
    file_st,file_end= text.split('-')



    for number in range(int(file_st), int(file_end)):
        return_dict = {}
        para, list_sentence = get_related_sub_verb_in_onetxt(files, number, list_tells,postagger,parser)
        if para == None:
            return json.dumps('数据有误！',ensure_ascii=False)
        print('para', para)
        print('list_sentence', list_sentence)

        for sentence in list_sentence:
            match_txt_ = match_txt(sentence, para)
            regulation_match_ = regulation_match(match_txt_)
            #count_ = 0
            if regulation_match_ != []:
                print('regulation_match_;', regulation_match_)
            results = compare_txt_similar(regulation_match_, 0.5, model)

            #results = filter(lambda x)
            if results != []:
                #print('results:', results)
                for x in results:
                    if x[0][0] in return_dict.keys():
                        return_dict[x[0][0]].append(''.join(x[1]))
                    else:
                        return_dict[x[0][0]]=[''.join(x[1])]
        list_sentence.append('-------------------------------------------------------------')
        return_dict['--------------------------------content------------------------------------------------------------------------------']= list_sentence
        print(return_dict)
    return json.dumps(return_dict,ensure_ascii=False)


if __name__=='__main__':
    #news_file_cut_word()
    r_ = open('said_txt_latest.json', 'r')
    print(r_)
    list_tell = json.load(r_)
    list_tells = [word[0] for word in list_tell if word[1]>=0]
    model = Word2Vec.load('C:\project-1\word_vector_model\\test.model')
    files = pd.read_csv('news_chinese.csv')
    par_model_path = 'C:\project-1\ltp_data_v3.4.0\parser.model'
    pos_model_path = 'C:\project-1\ltp_data_v3.4.0\pos.model'
    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型
    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型
    app.config['JSON_AS_ASCII'] = False
    app.run('0.0.0.0', port=31002, threaded=True, debug=True)
    start()






