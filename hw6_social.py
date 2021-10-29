"""
Social Media Analytics Project
Name:
Roll Number:
"""

from pandas.core.frame import DataFrame
import hw6_social_tests as test

project = "Social" # don't edit this

### PART 1 ###

import pandas as pd
import nltk
from typing import Counter 
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]

'''
makeDataFrame(filename)
#3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    df=pd.read_csv(filename)
    return df


'''
parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseName(fromString):
    for line in fromString.split("\n"):
        start=line.find(" ") 
        line=line[start:]
        end=line.find("(")
        line=line[:end]
        line=line.strip()
    return line


'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    for line in fromString.split("\n"):
        start=line.find("(") +len("(")
        line=line[start:]
        end=line.find("from")
        line=line[:end]
        line=line.strip()
    return line


'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseState(fromString):
    for line in fromString.split("\n"):
        start=line.find("from") + len("from")
        line=line[start:]
        end=line.find(")")
        line=line[:end]
        line=line.strip()
    return line


'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    lst=[] 
    m=message.split("#") 
    for x in m[1:len(m)]: 
        string=""
        for y in x: 
            if y not in endChars: 
                string+=y  
            else: 
                break 
        string="#"+string 
        lst.append(string) 
    # print(len(lst))
    return lst



'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    row= stateDf.loc[stateDf['state'] == state, 'region']
    #print(row)
    # print(stateDf)
    # print(row.values[0])
    return row.values[0]
    # print(row.values[0])


'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    names=[]
    positions=[]
    states=[]
    regions=[]
    hashtags=[]
    for index, row in data.iterrows():
        labelvalue=data["label"].loc[index]
        name=parseName(labelvalue)
        position=parsePosition(labelvalue)
        state=parseState(labelvalue)
        region=getRegionFromState(stateDf, state)
        textvalue=data["text"].loc[index]
        hashtag=findHashtags(textvalue)
        names.append(name)
        positions.append(position)
        states.append(state)
        regions.append(region)
        hashtags.append(hashtag)
    data["name"]=names
    data["position"]=positions
    data["state"]=states
    data["region"]=regions
    data["hashtags"]=hashtags
    # print(len(hashtags))
    return None



    return


### PART 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if score<-0.1:
        return "negative"
    elif score>0.1:
        return "positive"
    else:
        return "neutral"




'''
addSentimentColumn(data)
#2 [Check6-2]
Parameters: dataframe
Returns: None
'''
def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    sentiments=[]
    for index, row in data.iterrows():
        message=data["text"].loc[index] 
        text=findSentiment(classifier, message) 
        sentiments.append(text) 
    data["sentiment"]=sentiments
    return


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    count={} 
    if len(colName) !=0 and len(dataToCount) !=0 :
        for index, row in data.iterrows(): 
            if (row[colName] == dataToCount) : 
                if (row["state"] not in count) : 
                    count[row["state"]] = 0
                count[row["state"]] +=1
    elif colName=="" and dataToCount=="":
        for index, row in data.iterrows(): 
            if (row["state"] not in count) : 
                count[row["state"]] = 0
            count[row["state"]] +=1
    return count


'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    region_count={}
    for i,row in data.iterrows():
        region=row["region"]
        if region not in region_count:
            region_count[region] = {}
        if  region  in region_count:
            att=row[colName]
            if att not in region_count[region]:
                region_count[region][att] = 0
            region_count[region][att] += 1
    return region_count


'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    dict_={}
    for i in data["hashtags"]:
        for j in i:
            if len(j)!=0 and j not in dict_:
                dict_[j]=1
            else:
                dict_[j]+=1
    return dict_


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    hashtagsort={}
    counts=0
    hashtag_sorted_dict = sorted(hashtags, key=hashtags.get, reverse=True)
    for i in hashtag_sorted_dict:
        if counts<count:
            hashtagsort[i]= hashtags[i]
            counts=counts+1
    #print(hashtagssorted)
    return (hashtagsort)


'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    lst=[]
    for index,row in data.iterrows():
        if hashtag in row['text']:
            if row['sentiment']=='positive':
                lst.append(1)
            elif row['sentiment']=='negative':
                lst.append(-1)
            elif row['sentiment']=='neutral':
                lst.append(0)
    return sum(lst)/len(lst)


### PART 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    x=[i for i in stateCounts]
    w=0.8
    y=[stateCounts[i] for i in stateCounts]
    for index in range(len(x)):
        plt.bar(x[index],y[index],width=w)
    plt.xticks(ticks=list(range(len(x))),label=x,rotation="vertical")
    plt.title("StateCount")
    plt.xlabel("State")
    plt.ylabel("Count")
 
    plt.show()
    return



'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    featurerate={}
    topstates={}
    for i in stateFeatureCounts:
        featurerate[i]=(stateFeatureCounts[i]/stateCounts[i])
    topstates=dict(Counter(featurerate).most_common(n))
    graphStateCounts(topstates,"Top n Featured")
    return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    featurelist=[]
    regionlist=[]
    regionvalue=[]
    for i in regionDicts:
        temporarylist=[]
        x=regionDicts[i]
        for j in x:
            if j not in featurelist:
                featurelist.append(j)
            temporarylist.append(x[j])
        regionvalue.append(temporarylist)
        regionlist.append(i)
    sideBySideBarPlots(featurelist, regionlist, regionvalue, title)
    return


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    return


#### PART 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    # print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    # test.week1Tests()
    # print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    # test.runWeek1()
    # test.testParseName()
    # test.testParsePosition()
    # test.testParseState()
    #test.testFindHashtags()
    # test.testGetRegionFromState()
    # test.testAddColumns()
    # test.testAddSentimentColumn()
    # df = makeDataFrame("data/politicaldata.csv") 
    # stateDf = makeDataFrame("data/statemappings.csv") 
    # addColumns(df, stateDf)  
    # addSentimentColumn(df)
    #test.testGetDataCountByState(df)
    # test.testGetDataForRegion(df)
    # test.testGetHashtagRates(df)
    #test.testMostCommonHashtags(df)
    #test.testGetHashtagSentiment(df)

    

    ## Uncomment these for Week 2 ##
    """print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()"""

    ## Uncomment these for Week 3 ##
    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
