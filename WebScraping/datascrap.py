import nltk
import urllib
import bs4 as bs
import re
from nltk.corpus import stopwords
nltk.download('punkt_tab')
nltk.download('stopwords')

source=urllib.request.urlopen('https://en.wikipedia.org/wiki/Albert_Einstein').read()

soup= bs.BeautifulSoup(source,'lxml')

text=""
for paragraph in soup.find_all('p'):
    text+=paragraph.text
text=re.sub(r'\[[0-9]*\]',' ',text)
text=re.sub(r'\s+',' ',text)
text=text.lower()
text=re.sub(r'\d',' ',text)
text=re.sub(r'\s+',' ',text)

sentences=nltk.sent_tokenize(text)

sentences=[nltk.word_tokenize(sentence) for sentence in sentences]
print(sentences)
