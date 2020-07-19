import re
import jieba
import os
import numpy
import random
import time
import matplotlib.pyplot as plt

def fmt_newindex(filename):
    f = open(filename, 'r', encoding="utf-8")
    newindex = f.read().split("\n")
    f.close()
    if newindex[-1] == "":
        newindex.pop()
    for line in range(len(newindex)):
        newindex[line] = newindex[line].split(" ")
    return newindex

def getStopWords(filename):
  stopList=[]
  for line in open(filename, "r", encoding="utf-8"):
      stopList.append(line[:len(line)-1])
  return stopList

def get_wordList(content, wordsList, stopList):
    rule = re.compile(r"[^\u4e00-\u9fa5]")
    content = rule.sub("", content)
    res_list = list(jieba.cut(content))
    for word in res_list:
        if word not in stopList and word.strip() != '' and word != None:
            wordsList.append(word)
def addToDict(wordsList, spam_or_ham_Dict):
    for item in wordsList:
        if item in spam_or_ham_Dict.keys():
            spam_or_ham_Dict[item] += 1
        else:
            spam_or_ham_Dict.setdefault(item, 1)

def CalBayes(wordsList, spamDict, hamDict, spam_word_num, ham_word_num):
    spam_prob = 1
    ham_prob = 1
    for word in wordsList:
        if word in spamDict:
            spam_prob *= (spamDict[word] / spam_word_num)
        else:
            spam_prob *= 1     
        if word in hamDict:
            ham_prob *= (hamDict[word] / ham_word_num)
        else:
            ham_prob *= 1
    if(spam_prob >= ham_prob):
        return 1
    else:
        return 0

def test_spam_classify():
    newindex_name = "newindex"
    stopWords_name = "List.txt"

    spamDict = {}
    hamDict  = {}   

    wordsList = []  

    testResult = {}  

    ham_path_List  = []
    spam_path_List = []  
    test_path_List = []

    begin_time = time.time()

    All_List = fmt_newindex(newindex_name)
    Train_List = All_List[:(int)(len(All_List) * 0.7)]
    Test_List  = All_List[(int)(len(All_List) * 0.7):len(All_List)]
    for i in Train_List:  
        if i[0][0] == 'h':
            ham_path_List.append(i[1])
        else:
            spam_path_List.append(i[1])

    ham_num = len(ham_path_List)
    spam_num = len(spam_path_List)

    stopList = getStopWords(stopWords_name)


    for ham_path in ham_path_List:
        wordsList.clear()
        for line in open(ham_path, "r"):
            get_wordList(line, wordsList, stopList)
        ham_word_num = len(wordsList)
        addToDict(wordsList, hamDict)


    for spam_path in spam_path_List:
        wordsList.clear()
        for line in open(spam_path, "r"):
            get_wordList(line, wordsList, stopList)
        spam_word_num = len(wordsList)
        addToDict(wordsList, spamDict)


    for Test_email in Test_List:
        wordsList.clear()
        for line in open(Test_email[1], "r"):
            get_wordList(line, wordsList, stopList)
        select = CalBayes(wordsList, spamDict, hamDict, spam_word_num, ham_word_num)
        if(select == 1):
            testResult.setdefault(Test_email[1], 'spam')
        else:
            testResult.setdefault(Test_email[1], 'ham')

    end_time = time.time()

    f2 = open('Result.txt', encoding='utf-8', mode='w')
    right_number = 0

    plot_x_list = range(1, int(len(All_List)*0.3)+1)
    plot_y_list = [-1] * int(len(All_List)*0.3)

    for i in range(int(len(All_List)*0.3)):
        if Test_List[i][0] == testResult[Test_List[i][1]]:
            right_number += 1
            plot_y_list[i] = 1
        f2.write("Number " + str(i) + " Test_email: The path is " + Test_List[i][1] +
        ", The real label is " +Test_List[i][0] + ", The Naive_bayes label is " + testResult[Test_List[i][1]] + '\n')
    accuracy = right_number / (len(All_List) * 0.3)
    spam_word_rank = sorted(spamDict.items(), key=lambda d: d[1], reverse=True)[0:20]
    f2.write("\nThe most 20 frequency word in Spam are: ")
    for word in spam_word_rank:
        f2.write(word[0] + " ")
    f2.write("\n")
    ham_word_rank = sorted(hamDict.items(), key=lambda d: d[1], reverse=True)[0:20]
    f2.write("\nThe most 20 frequency word in Ham are: ")
    for word in ham_word_rank:
        f2.write(word[0] + " ")
    f2.write("\n")
    f2.write("\nThe total Test_number is %d, the right_number is %d, the accuracy is %f \n"
             % (len(All_List)*0.3, right_number, accuracy))
    f2.close()
    print("The total Test_number is %d, the right_number is %d, the accuracy is %f \n"
          % (len(All_List)*0.3, right_number, accuracy))
    print("The total spam_detection time is %d s" % (end_time-begin_time))

    plt.title('The right or wrong detection(1 is right and -1 is wrong)')  
    plt.xlabel('Test_email (accuracy is  14523/15320 * 100%)')
    plt.ylabel('right or wrong')
    plt.xlim(0, len(testResult) + 1)
    plt.ylim(-1.5, 1.5)
    plt.scatter(plot_x_list, plot_y_list, marker='o', color='red', s=0.0004, label='Test_Email')
    plt.legend(loc='best')  
    plt.show()

if __name__ == "__main__":
    test_spam_classify()