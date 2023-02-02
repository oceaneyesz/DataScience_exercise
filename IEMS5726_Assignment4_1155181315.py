# <1155181315>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import string
from wordcloud import WordCloud
import matplotlib.ticker as mticker

# Problem 2
def problem_2(filenames):
    # write your logic here
    ymin, ymax = 0, 100
    csv_data_A = []
    csv_data_B = []
    for i in range(len(filenames)):
        csv_data = pd.read_csv(filenames[i])
        csv_data_A.append(list(csv_data.iloc[:, 0]))
        csv_data_B.append(list(csv_data.iloc[:, 1]))
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.boxplot(csv_data_A)
    ax1.set_title('Test 1')
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Score")
    ax1.grid()
    ax1.set_ylim([ymin, ymax])
#     ax1.set_xticks([1,2,3])
    ax1.set_xticklabels(['Class A', 'Class B', 'Class C'])
    ax2 = fig.add_subplot(gs[0, 1])
#     ax2.set_xticks([1,2,3])
    ax2.set_xticklabels(['Class A', 'Class B', 'Class C'])
    ax2.set_ylim([ymin, ymax])
    ax2.boxplot(csv_data_B)
    ax2.set_ylabel('Score')
    ax2.set_xlabel("Class")
    ax2.set_title('Test 2')
    ax2.set_ylim([ymin, ymax])
    ax2.grid()
    
    plt.suptitle("Test result")
    plt.savefig("problem2")

import re
# from wordcloud import WordCloud
# Problem 3
def problem_3(filenames):
    # write your logic here
    gs = gridspec.GridSpec(len(filenames),2)
    fig = plt.figure()
    for i in range(len(filenames)):
        file = open(filenames[i],'rb')
        fig.add_subplot(gs[int(i/2), i%2])
        data = file.read()
        file.close( )
        punctuation_string = string.punctuation
        data = data.decode()
        for i in punctuation_string:
            data = data.replace(i, '')
        data = data.lower()
        data = ''.join([i for i in data if not i.isdigit()])
        re.sub(r'[^\w]', ' ', data)
        word_cloud = WordCloud(collocations = False, background_color = 'white', random_state=5726).generate(data)
        plt.axis('off')
        plt.imshow(word_cloud)
    plt.savefig("problem3")

# Problem 4
def problem_4(filename,start,end,target):
    # write your logic here
    csv_data = pd.read_csv(filename)
    tmp_csv = []
#     csv_data_name = csv_data[csv_data['Name'].str.contains(target)]
#     for i in range(len(target)):
#         tmp_csv = [tmp_csv csv_data.loc[csv_data['Name']==target[i]]]
#     csv_data = pd.concat(tmp_csv)
    csv_data['date'] = pd.to_datetime(csv_data['date'])
    to_start = pd.to_datetime(start)
    to_end = pd.to_datetime(end)
#    csv_data = csv_data.loc[(csv_data['Name']==target[0]) | (csv_data['Name']==target[1]) | (csv_data['Name']==target[2])]
    csv_data = csv_data.loc[(csv_data['date'])>=start]
    csv_data = csv_data.loc[(csv_data['date'])<=end]
    csv_data.dropna()
    print(csv_data['date'].iloc[-1:])
    
    for i in range(len(target)):
        x = csv_data.loc[csv_data['Name'] == target[i]]['date']
        y = csv_data.loc[csv_data['Name'] == target[i]]['close']
        plt.plot(x,y,label=target[i])
#         plt.text(csv_data['date'].iloc[-1:], -10, target[i])
        plt.text(x.iloc[-1:], y.iloc[-1:], target[i])
    plt.title("Close value")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.grid()
#     plt.legend(target)
    plt.savefig("problem4") # do not call plt.show()
    

# Problem 5
def problem_5(df):
    # write your logic here
    font={'size':8}
    sum1 = (df.iloc[0,0]+df.iloc[0,1])
    sum2 = (df.iloc[1,0]+df.iloc[1,1])
    df.iloc[0,0] = df.iloc[0,0]/sum1
    df.iloc[0,1] = df.iloc[0,1]/sum1
    df.iloc[1,0] = df.iloc[1,0]/sum2
    df.iloc[1,1] = df.iloc[1,1]/sum2
    df.plot.barh(stacked=True, title='Passing Percetage',figsize = (12, 8))
    plt.ylabel("Years")
    t1 = "{:.4f}".format(df.iloc[0,0]/(df.iloc[0,0]+df.iloc[0,1]))
    t2 = "{:.4f}".format(df.iloc[0,1]/(df.iloc[0,0]+df.iloc[0,1]))
    t3 = "{:.4f}".format(df.iloc[1,1]/(df.iloc[1,0]+df.iloc[1,1]))
    t4 = "{:.4f}".format(df.iloc[1,0]/(df.iloc[1,0]+df.iloc[1,1]))
    plt.text(df.iloc[0,0], 0, t1,fontdict=font)
    plt.text(df.iloc[0,0]+df.iloc[0,1], 0, t2,fontdict=font)
    plt.text(df.iloc[1,0], 1, t4,fontdict=font)
    plt.text(df.iloc[1,0]+df.iloc[1,1], 1, t3,fontdict=font)
    plt.legend(loc="upper right")
#     plt.rcParams['figure.figsize'] = (8.0, 4.0)
    plt.savefig("problem5",dpi=500) # do not call plt.show()

# Problem 6
def problem_6(filename,start,end,column):
    # write your logic here
    csv_data = pd.read_csv(filename)
    csv_data.dropna()
    csv_data['Date'] = pd.to_datetime(csv_data['Date'], format="%d/%m/%Y")
    data_grouped = csv_data.groupby("Date").agg({"Temperature": "mean",
                                                           "Fuel_Price": "mean",
                                                           "CPI": "mean",
                                                           "Unemployment": "mean",
                                                           "IsHoliday": "min"}).reset_index().fillna(method='pad', axis=0)
    data_grouped.head()
    fig = plt.figure(figsize = (16, 12))
    count_columns_ex_date = len(data_grouped.columns[1:])
    for idx, col in enumerate(data_grouped.columns[1:]):
        plt.subplot(count_columns_ex_date, 1, idx+1)
        plt.plot(data_grouped["Date"], data_grouped[col])
        plt.ylabel(col)

    plt.savefig("problem6") # do not call plt.show()
    
# filenames = ["classA.csv","classB.csv","classC.csv"]
# problem_2(filenames)
# filenames = ["paragraph1.txt","paragraph2.txt","paragraph3.txt"]
# problem_3(filenames)
# start = "1/1/2018"
# end = "14/1/2018"
# target = ["ABBV", "AIV", "DFS"]
# problem_4("all_stocks_5yr.csv",start,end,target)
# students = pd.DataFrame({'Boys': [67, 78], 'Girls': [72, 80], },index=['First Year', 'Second Year'])
# problem_5(students)
# file = "Features data set.csv"
# start = "1/1/2010"
# end = "31/7/2013"
# col = ["Temperature","Fuel_Price","CPI","Unemployment","IsHoliday"]
# problem_6(file,start,end,col)
