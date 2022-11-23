import numpy as np
import nltk
import spacy
import requests
from bs4 import BeautifulSoup
nltk.download('punkt')
import re
import heapq
from numpy.linalg import norm
from collections import Counter
import pandas as pd
from nltk.tokenize import MWETokenizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize

nlp=spacy.load("en_core_web_sm")
url = ["https://en.wikipedia.org/wiki/Natural_language_processing", "https://www.gyansetu.in/what-is-natural-language-processing/", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8120048/", "https://languagelog.ldc.upenn.edu/nll/?p=2946", "https://www.fcg-net.org/"]
  
def texto(x):
  result = requests.get(x).text
  soup = BeautifulSoup(result, 'lxml')

  text = "".join([i.text for i in soup.find_all('p')])
  return text

def sentencas(text):
  return sent_tokenize(text)



def main():
  lista = []
  for z in url:
    lista2 = sentencas(texto(z))
    lista += lista2
  

  # print(lista)

  corpus = " ".join(lista)
  z = nltk.sent_tokenize(corpus)
  vocab = vocabulario(corpus)
  listaTfidf = []
  contagem = []
  for x in lista:
    contagem = vetor(x, vocab)
    tfValor = tf(contagem, x.split())
    idfValor = idf(lista, contagem)
    tfidfValor = tfidf(tfValor, idfValor)
    listaTfidf.append(tfidfValor)
    cosseno(tfidfValor, tfidfValor)
    
  matrizSimilaridade(vocab, listaTfidf)

  # idf(lista, contagem)


def vocabulario(corpus):
  palavras = corpus.split()
  seen = set()
  result = []
  for item in palavras:
      if item not in seen:
          seen.add(item)
          result.append(item)

  # print("Vocabulario: ", result)
  return result

def vetor(lista, vocabulario):
  contador = []

  tokenizer = MWETokenizer()
  result = tokenizer.tokenize(word_tokenize(lista))

  fdist = FreqDist(result)
  # print(fdist.keys())
  lista = list(fdist.keys())

  for x in vocabulario:
    contador.append(fdist[x])

  # print(f"Frequencia: {contador}")
  return contador


def tf(frequencia, sentenca):
  tf_result = []
  bowCount = len(sentenca)
  for count in frequencia:
    tf_result.append(round(count/float(bowCount), 2))

  # print(f'TF: {tf_result}')
  return tf_result

def idf(docList, frequencia):
  import math
  idfLista = []
  n = len(docList)

  # frequencia = [i for i in frequencia if i != 0]

  for count in frequencia:
    if count != 0:
      idfLista.append(round(math.log10(n /float(count)), 3))
    else:
      idfLista.append(0)

  # print(f'IDF: {idfLista}')
  return idfLista

def tfidf(tfBow, idfs):
  tf_idf = []
  for val, val2 in zip(tfBow, idfs):
    tf_idf.append(round((val*val2), 3))

  # print(f'TFIDF: {tf_idf}')
  return tf_idf

def cosseno(a, b):
  return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def matrizSimilaridade(vocabulario, tfidf):
  df = pd.DataFrame(columns= range(1, len(tfidf)), index=range(1, len(tfidf)))

  for i in df.index:
    for j in df.columns:
     df.loc[i][j] = cosseno(tfidf[i-1], tfidf[j-1])

  display(df)

  
main()
