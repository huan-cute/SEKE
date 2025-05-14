import os
import re
import stanfordcorenlp
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import json
import nltk
from nltk.stem import  PorterStemmer

nlp = stanfordcorenlp.StanfordCoreNLP(r'../CoreNLP/stanford-corenlp-4.5.4')

#去除驼峰命名
def split_camel_case(s):
    #将字符串s切分为单词列表
    words = s.split()
    for index,word in enumerate(words):
        splitted = ' '.join(re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))',word))
        if splitted:
            words[index] = splitted

        return ' '.join(words)

#分词
def tokenize_text(text):
    return word_tokenize(text)

#将词转化为小写
def words_to_lowercase(words):
    return [word.lower() for word in words]

#去除停用词，标点和数字
def filter_words(word_list):
    stop_words = set(stopwords.words('english'))
    punctuation_symbols = set(string.punctuation)
    words = []
    for word in word_list:
        if '.' in word:
            for each in word.split('.'):
                words.append(each)
        words.append(word)
        filtered_words = [word for word in words if word.lower() not in stop_words and
                        not any(char in punctuation_symbols for char in word) and word.isalpha()]
        return filtered_words

#词形还原
def extract_restore(text):
    stemmer = PorterStemmer()

    def split_text(text,max_length):
        return [text[i:i + max_length] for i in range(0,len(text),max_length)]

    chunks = split_text(text,100000)
    all_stems = []
    for chunk in chunks:
        doc = nlp.annotate(chunk,properties={
            'annotators':'lemma',
            'pipelineLanguage':'en',
            'outputFormat':'json'
        })
        doc = json.loads(doc)
        lemmas = [word['lemma'] for sentence in doc['sentences'] for word in sentence['tokens']]
        stems = [stemmer.stem(token) for token in lemmas]
        all_stems.extend(stems)
    return  ' '.join(all_stems)

#预处理
def preprocessing(dataset_name):
    file_name_cc = os.listdir('../dataset/cc/' + dataset_name)
    file_name_uc = os.listdir('../dataset/uc/' + dataset_name)
    open('../dataset_output/cc' + dataset_name + '.txt','w').close()
    open('../dataset_output/uc' + dataset_name + '.txt','w').close()
    for file_name in file_name_cc:
        with open('../dataset/cc/' + dataset_name + '/' + file_name,'r',encoding='ISO-8859-1')as cf:
            text=""
            lines = cf.readlines()
            for line in lines:
                text += line.strip()+' '
            res = split_camel_case(text)
            res = tokenize_text(res)
            res = words_to_lowercase(res)
            res = filter_words(res)
            res = extract_restore(' '.join(res))
        with open('../dataset/cc/'+dataset_name+'.txt','a',encoding='ISO-8859-1')as cwf:
            cwf.write(res)
            cwf.write('\n')
        for file_name in file_name_uc:
            with open('../dataset/uc/'+dataset_name+'/'+file_name,'r',encoding='ISO-8859-1')as uf:
                text=""
                lines = uf.readlines()
                for line in lines:
                    text += line.strip()+' '
                res = split_camel_case(text)
                res = tokenize_text(res)
                res = words_to_lowercase(res)
                res = filter_words(res)
                res = extract_restore(' '.join(res))
            with open('../dataset_output/uc/' + dataset_name + '.txt','a',encoding='ISO-8859-1')as uwf:
                uwf.write(res)
                uwf.write('\n')



preprocessing('Drools')






















