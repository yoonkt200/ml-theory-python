```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import re
import ast


# 뉴스기사 리스트 크롤링
base_url = 'http://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid1=100&sid2=269'
req = requests.get(base_url)
html = req.content
soup = BeautifulSoup(html, 'lxml') # pip install lxml
newslist = soup.find(name="div", attrs={"class":"newsflash_body"})
newslist_atag = newslist.find_all('a')
url_list = []
for a in newslist_atag:
    url_list.append(a.get('href'))
    

# 텍스트 정제 함수
def text_cleaning(text):
    result_list = []
    for item in text:
        cleaned_text = re.sub('[a-zA-Z]', '', item)
        cleaned_text = re.sub('[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]',
                          '', cleaned_text)
        result_list.append(cleaned_text)
    return result_list

def removeNumberNpunct(doc):
    text = ''.join(c for c in doc if c.isalnum() or c in '+, ')
    text = ''.join([i for i in text if not i.isdigit()])
    return text


# 각 기사에서 텍스트만 정제하여 추출
req = requests.get(url_list[0])
html = req.content
soup = BeautifulSoup(html, 'lxml')
text = ''
doc = None
for item in soup.find_all('div', id='articleBodyContents'):
    text = text + str(item.find_all(text=True))
    text = ast.literal_eval(text)
    doc = text_cleaning(text[9:])
    
word_corpus = (' '.join(doc))
word_corpus = removeNumberNpunct(word_corpus)


# 텍스트에서 형태소 추출 -> pip install konlpy, jpype1, Jpype1-py3
from konlpy.tag import Twitter
from collections import Counter

nouns_tagger = Twitter()
nouns = nouns_tagger.nouns(word_corpus)
count = Counter(nouns)


# 형태소 워드 클라우드로 시각화 -> pip install pytagcloud, webbrowser
# Mac OS : /anaconda/envs/fastcampus/lib/python3.6/site-packages/pytagcloud/fonts
# Windosw OS : C:\Users\USER\Anaconda3\envs\pc36 (가상환경주소) \Lib\site-packages\pytagcloud\fonts
# 위 경로에 NanumBarunGothic.ttf 파일 옮기기
import random
import pytagcloud
import webbrowser

ranked_tags = count.most_common(40)
taglist = pytagcloud.make_tags(ranked_tags, maxsize=80)
pytagcloud.create_tag_image(taglist, 'wordcloud.jpg', size=(900, 600), fontname='Korean', rectangular=False)
```