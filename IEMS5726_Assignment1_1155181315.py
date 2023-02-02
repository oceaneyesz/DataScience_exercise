# <1155181315>
import numpy as np
import pandas as pd
from scipy.optimize import leastsq
import scipy.stats

def problem_2(a,b):
    output = np.dot(a,b)
    return output

def problem_3(a,b):
    output = np.outer(a,b)
    return output

# Problem 4
def problem_4(a,b):
    output = np.multiply(a, b)
    return output

# Problem 5
def problem_5(filename, col):
    # write your logic here, df is a dataframe, instead of number
    df = pd.read_csv(filename)
    df = df.dropna(axis=0,how='any')
    df.reset_index(drop=True, inplace=True)
    df = df[col]
    return df

# Problem 6
def problem_6(filename, threshold):
    # write your logic here, df is a dataframe, instead of number
    df = pd.read_csv(filename)
    df = df.fillna(df.mean())
    df = df[(df['AGE'] > threshold['AGE']) & (df['INDUS'] > threshold['INDUS']) & (df['MEDV'] > threshold['MEDV'])]
    df = df[['AGE','INDUS','MEDV']]
    df.reset_index(drop=True, inplace=True)
    df = df.sort_values(by = ['AGE', 'INDUS', 'MEDV'])
    return df

# Problem 7
def problem_7(filename, n, col, threshold):
    # write your logic here, df is a dataframe, instead of number
    df = pd.read_csv(filename)
    df = df[::n]
    df = df[col]
    df = df.dropna(axis=0,how='any')
    df['CRIM'][df['CRIM'] > threshold] = 'high'
    df['CRIM'][df['CRIM'] != 'high'] = 'low'
    meanhigh = df['DIS'][df['CRIM'] == 'high'].mean()
    meanlow = df['DIS'][df['CRIM'] == 'low'].mean()
    return df, meanhigh, meanlow


# Problem 8
def problem_8(df1, df2):
    # write your logic here, k is an array, instead of a number
    A0 = df1['A'].values
    A1 = df1['B'].values
    A2 = np.vstack([A0, A1])
    A = np.vstack([A2, np.ones(len(A0))]).T
    y = np.array(df2)
    k1, k2, b = np.linalg.lstsq(A, y, rcond=None)[0]
    k = [k1[0], k2[0]]
    b = b[0]
    return k, b

# Problem 9
def problem_9(df):
    # write your logic here
    x = df['A'].values
    y = df['B'].values
    s = scipy.stats.spearmanr(x, y)[0]
    p = scipy.stats.pearsonr(x, y)[0]
    return s, p

