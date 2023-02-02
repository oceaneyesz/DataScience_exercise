# 1155181315
import numpy as np
import pandas as pd
import csv
import nltk
import string
from nltk.stem.porter import *
import cv2 as cv
import matplotlib.pyplot as plt
import torchaudio
from torchaudio import transforms
from numpy import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets
from nltk.text import TextCollection
from nltk.tokenize import word_tokenize
import torch
import math

# Problem 2
def problem_2(filename,name):
    # write your logic here, df is a dataframe
    df = 0
    csv_data = pd.read_csv(filename)
    dummies = pd.get_dummies(csv_data[name])
    dummies.drop(dummies.columns[[-1]], axis = 1, inplace = True)
    df = csv_data.join(dummies)
    return df

# Problem 3
def problem_3(filename,k):
    # write your logic here, pc is a numpy array
    np.set_printoptions(suppress=True)
    data = pd.read_csv(filename)
    X = StandardScaler().fit_transform(data)
    pca = PCA(n_components=k, whiten=True)
    X_pca = pca.fit_transform(X)
    pc = np.array(X_pca)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    return pc

# Problem 4
def problem_4(sentence):
    # write your logic here

    punctuation_string = string.punctuation
    for i in punctuation_string:
        sentence = sentence.replace(i, '')
    sentence = sentence.lower()
    words = nltk.tokenize.word_tokenize(sentence)
    porter_stemmer = PorterStemmer()
    words = [porter_stemmer.stem(a) for a in words]
    bigrams = list(nltk.ngrams(words, 2))
    bigrams = [' '.join(x) for x in bigrams]
    output = words + bigrams
    return output

# Problem 5
def problem_5(doc):
    # write your logic here, df is a dataframe, instead of number
    words = []
    lists = []
    ans = []
    for i in doc:
        words.append(problem_4(i))
        for j in problem_4(i):
            lists.append(j)
    lists = sorted(list(set(lists)))
    for word in lists:
        for i in range(len(words)):
            w = tf_idf(word, words[i], words)
            ans.append(w)
    step = len(words)
    ans_list = []
    ans_list = [ans[i:i+step] for i in range(0, len(ans),step)]
    ans_list = list(map(list, zip(*ans_list)))
    df = pd.DataFrame(data = ans_list, columns = lists)
    return df

def tf(word, word_dict):
    return word_dict.count(word)
    #print(word_dict)
def dfw(word,word_list):
    w = 0
    for i in word_list:
        if word in i:
            w +=1
    return w
def tf_idf(word,word_dict,word_list):
    if dfw(word,word_list) == 0:
        return 0
    else:
        return round(tf(word, word_dict) * (math.log(len(word_list) / (dfw(word, word_list)),10)),3)

# Problem 6
def problem_6(image_filename):
    # write your logic here, keypoint and descriptor are BRISK object
    image1 = cv.imread(image_filename,flags = cv.IMREAD_GRAYSCALE)
    BRISK = cv.BRISK_create()
    keypoint, descriptor = BRISK.detectAndCompute(image1, None)

    return keypoint, descriptor

# Problem 7
def problem_7(image1_filename, image2_filename):
    # write your logic here, common_descriptor is the common desc.
    keypoints1, descriptors1 = problem_6(image1_filename)
    keypoints2, descriptors2 = problem_6(image2_filename)
    BFMatcher = cv.BFMatcher(normType = cv.NORM_HAMMING,crossCheck = True)
    matches = BFMatcher.match(queryDescriptors = descriptors1,trainDescriptors = descriptors2)
    common_descriptor = sorted(matches, key = lambda x: x.distance)
    return common_descriptor

# Problem 8
def problem_8(audio_filename, sr, n_mels, n_fft):
    # write your logic here, spec is a tensor
    sig, old_sr = torchaudio.load(audio_filename)
    num_channels = sig.shape[0]
    resig = torchaudio.transforms.Resample(old_sr, sr)(sig[:1,:])
    if (num_channels > 1):
        retwo = torchaudio.transforms.Resample(old_sr, sr)(sig[1:,:])
        resig = torch.cat([resig, retwo])
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, n_mels=n_mels)(resig)
    return spec

# Problem 9
def problem_9(spec, max_mask_pct, n_freq_masks, n_time_masks):
    # write your logic here
    a, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec
    freq_mask_param = max_mask_pct * n_mels
    for a in range(n_freq_masks):
        aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
    time_mask_param = max_mask_pct * n_steps
    for a in range(n_time_masks):
        aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
    return aug_spec