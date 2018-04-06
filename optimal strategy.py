import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import xgboost as xgb

def import_dataset():
    
    train = pd.read_csv("G:/Business Analytics/Web Economics/code for we/train.csv")
    validation = pd.read_csv("G:/Business Analytics/Web Economics/code for we/validation.csv")
    
    return train, validation

def dataset_info():

    train.info()
    train.describe()
    train.click.value_counts()
    train.click.value_counts() / train.click.shape[0]
    return 0     

def constant_bid(validation): 
    output1 = []
    for price in range(1,300):
        Temp = validation[["click","payprice"]][validation.payprice < price ]
        paid = 0
        click = 0
        impression = 0
        budget = 6250 * 1000
        
        Temp = Temp.sample(frac = 1)
        Temp1 = Temp.as_matrix()
        
        for i in range(1,len(Temp1)):
            if paid + Temp1[i][1] > budget :
                break
            else:
                paid = paid + Temp1[i][1]
                click = click + Temp1[i][0]
                impression = impression + 1
        output1.append([paid,click,click/(1*impression),paid/impression]) ##### output format: cost, click, CTR ######
    output = pd.DataFrame(output1, columns = ['cost','clicks','CTR','CPC'])
    output['bidprice'] = np.arange(1,300)
#   output.to_csv("G:/Business Analytics/Web Economics/code for we/constant_bidding.csv",index = False)
    return output


def random_bid():
    t = time.time()

    val = pd.read_csv("G:/Business Analytics/Web Economics/code for we/val.csv")
    val["bidprice"] = 0
    val = np.array(val)
    output1 = []
    
    for lower in range(50,250):
        for upper in range(lower + 1,301):
            valtemp = val.copy()
            for i in range(len(val)):
                valtemp[i][2] = np.random.randint(lower,upper)
                valtemp[i][1] = val[i][1] - valtemp[i][2]
                if valtemp[i][1] < 0:
                    valtemp[i][1] = 1
                else:
                    valtemp[i][1] = 0
            valtemp = pd.DataFrame(valtemp,columns = ['click','payrice','bidprice'])
            valtemp = valtemp.sample(frac = 1)
            valtemp = np.array(valtemp)
            
            paid = 0
            click = 0
            impression = 0
            budget = 6250 * 1000 
            for i in range(len(valtemp)):
                if paid + valtemp[i][1] * valtemp[i][2] > budget:
                    break
                else:
                    paid = paid + valtemp[i][1] * valtemp[i][2]
                    click = click + valtemp[i][0]* valtemp[i][1]
                    impression = impression + valtemp[i][1]
            output1.append([lower,upper,paid/1000,click,click/impression])
        
    output = pd.DataFrame(output1,columns = ['lowerbound','upperbound','paid','click','CTR'])
    #a1 = pd.read_csv("G:/Business Analytics/Web Economics/code for we/ran/random_bid1 (1).csv") # 50, 81, 109
    #a2 = pd.read_csv("G:/Business Analytics/Web Economics/code for we/ran/random_bid (1).csv") # 50, 55, 130
    #a3 = pd.read_csv("G:/Business Analytics/Web Economics/code for we/ran/random_bid1.csv") #82,122,47
    #a4 = pd.read_csv("G:/Business Analytics/Web Economics/code for we/ran/random_bid.csv") # 80, 87, 52
        
    #a1 = pd.read_csv("G:/Business Analytics/Web Economics/code for we/ran/random_bidding_1_50.csv") # 40, 99, 54
    #a2 = pd.read_csv("G:/Business Analytics/Web Economics/code for we/ran/random_bidding_50_100.csv") # 51, 94, 53
    #a3 = pd.read_csv("G:/Business Analytics/Web Economics/code for we/ran/random_bidding_100_150.csv") # 102, 105, 45
    #a4 = pd.read_csv("G:/Business Analytics/Web Economics/code for we/ran/random_bidding_150_200.csv") # 172, 174, 40
    #a5 = pd.read_csv("G:/Business Analytics/Web Economics/code for we/ran/random_bidding_200_300.csv") # 206, 274, 33
    return output

 

def recal(p,w):
    return p/(p + (1-p)/w)

def plt_constant_bid(output):
    fig, ax1 = plt.subplots() # 使用subplots()创建窗口
    ax2 = ax1.twinx()
    
    ax1.set_xlabel("bidding price")
    ax1.set_ylabel("clicks")
    ax2.set_ylabel("CTR")
    ax1.plot(output.click,c = 'blue',label = "clicks")
    ax2.plot(output.ctr,c = 'red',label = "CTR")
    plt.show()
    return

def drop_cols():
    column = [ 'click','weekday', 'hour', 'useragent', 'region', 'city', 'adexchange', 'slotwidth', 'slotheight', 'slotvisibility','slotformat', 'slotprice', 'bidprice','payprice', 'keypage', 'advertiser','usertag']
    train = pd.read_csv("G:/Business Analytics/Web Economics/train.csv")
    validation = pd.read_csv("G:/Business Analytics/Web Economics/validation.csv")
    test = pd.read_csv("G:/Business Analytics/Web Economics/test1.csv")
    
    train1 = train[column]
    validation1 = validation[column]
    test1 = test[[ 'weekday', 'hour', 'useragent', 'region', 'city', 'adexchange', 'slotwidth', 'slotheight', 'slotvisibility','slotformat', 'slotprice',  'keypage', 'advertiser','usertag']]
    
    train1.to_csv("G:/Business Analytics/Web Economics/code for we/train.csv",index = False)
    validation1.to_csv("G:/Business Analytics/Web Economics/code for we/validation.csv",index = False)
    test1.to_csv("G:/Business Analytics/Web Economics/code for we/test.csv",index = False)
    return 


def negative_down_sampling(train,ratio):
    pos = train[train.click == 1]
    neg = train[train.click == 0]
    neg = neg.sample(frac = ratio)
    a = pos.append(neg)
    a = a.sample(frac = 1)
    a = np.array(a)
    a = pd.DataFrame(a,columns = train.columns)
    return a
#a = pd.read_csv("G:/Business Analytics/Web Economics/code for we/train_neg_0.025.csv")
#a.to_csv("G:/Business Analytics/Web Economics/code for we/train_neg_0.025.csv",index = False)
def encoding1(XX): ## to encode the test dataset
    X = XX
    X = pd.concat([X,pd.get_dummies(X.weekday,prefix='day')],axis=1)
    X = X.drop('weekday',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.hour,prefix='hour')],axis=1)
    X = X.drop('hour',axis=1)
    
    df = pd.DataFrame(X.useragent.str.split('_',1).tolist(),columns = ['OS','browser'])
    X = pd.concat([X,df],axis=1)
    X = pd.concat([X,pd.get_dummies(X.OS,prefix='OS')],axis=1)
    X = X.drop('OS',axis=1)
    X = pd.concat([X,pd.get_dummies(X.browser,prefix='browser')],axis=1)
    X = X.drop('browser',axis=1)
    X = X.drop('useragent',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.region,prefix='region')],axis=1)
    X = X.drop('region',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.city,prefix='city')],axis=1)
    X = X.drop('city',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.adexchange,prefix='adexchange')],axis=1)
    X = X.drop('adexchange',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.slotwidth,prefix='slotwidth')],axis=1)
    X = X.drop('slotwidth',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.slotheight,prefix='slotheight')],axis=1)
    X = X.drop('slotheight',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.slotvisibility,prefix='slotvisibility')],axis=1)
    X = X.drop('slotvisibility',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.slotformat,prefix='slotformat')],axis=1)
    X = X.drop('slotformat',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.keypage,prefix='keypage')],axis=1)
    X = X.drop('keypage',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.advertiser,prefix='advertiser')],axis=1)
    X = X.drop('advertiser',axis=1)
    
    X = X.drop('slotprice',axis=1)
    
    
    a = pd.DataFrame(X.usertag.str.split(',').tolist())
    usertag_df = pd.DataFrame(a)
    usertag_df2 = pd.get_dummies(usertag_df,prefix='usertag')
    usertag_df2 = usertag_df2.groupby(usertag_df2.columns, axis=1).sum()
    X = pd.concat([X, usertag_df2], axis=1)
    X = X.drop('usertag', axis=1)
    
    return X

def encoding(XX):## to encode the train and validation dataset
    X = XX
    X = pd.concat([X,pd.get_dummies(X.weekday,prefix='day')],axis=1)
    X = X.drop('weekday',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.hour,prefix='hour')],axis=1)
    X = X.drop('hour',axis=1)
    
    df = pd.DataFrame(X.useragent.str.split('_',1).tolist(),columns = ['OS','browser'])
    X = pd.concat([X,df],axis=1)
    X = pd.concat([X,pd.get_dummies(X.OS,prefix='OS')],axis=1)
    X = X.drop('OS',axis=1)
    X = pd.concat([X,pd.get_dummies(X.browser,prefix='browser')],axis=1)
    X = X.drop('browser',axis=1)
    X = X.drop('useragent',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.region,prefix='region')],axis=1)
    X = X.drop('region',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.city,prefix='city')],axis=1)
    X = X.drop('city',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.adexchange,prefix='adexchange')],axis=1)
    X = X.drop('adexchange',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.slotwidth,prefix='slotwidth')],axis=1)
    X = X.drop('slotwidth',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.slotheight,prefix='slotheight')],axis=1)
    X = X.drop('slotheight',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.slotvisibility,prefix='slotvisibility')],axis=1)
    X = X.drop('slotvisibility',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.slotformat,prefix='slotformat')],axis=1)
    X = X.drop('slotformat',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.keypage,prefix='keypage')],axis=1)
    X = X.drop('keypage',axis=1)
    
    X = pd.concat([X,pd.get_dummies(X.advertiser,prefix='advertiser')],axis=1)
    X = X.drop('advertiser',axis=1)
    
    X = X.drop('slotprice',axis=1)
    X = X.drop('bidprice',axis=1)
    X = X.drop('payprice',axis=1)
    
    a = pd.DataFrame(X.usertag.str.split(',').tolist())
    usertag_df = pd.DataFrame(a)
    usertag_df2 = pd.get_dummies(usertag_df,prefix='usertag')
    usertag_df2 = usertag_df2.groupby(usertag_df2.columns, axis=1).sum()
    X = pd.concat([X, usertag_df2], axis=1)
    X = X.drop('usertag', axis=1)
    
    return X
def non_lin_1(val,c,lamba = 5.2e-7): ###best c = 2.6, click = 71 ##
    val.price = np.sqrt(c * val.pctr / lamba + c **2) - c
    val.got[val.price >= val.payprice] = 1
    val.got[val.price < val.payprice] = 0
    
    val1 = val.sample(frac = 1)
    val1 = np.array(val1)
    paid = 0
    click = 0
    impression = 0
    budget = 6250 * 1000
    for i in range(len(val1)):
        if paid + int(val1[i][3]) * val1[i][4] > budget:
            break
        else:
            paid = paid + int(val1[i][3]) * val1[i][4]
            click = click + val1[i][0] * val1[i][4]
            impression = impression + val1[i][4]
    output.append([c,paid, click, click/impression,paid/impression])
    return output

#for i in np.arange(9,11,0.1):
#    non_lin_2(val,i)
#output1 = pd.DataFrame(output,columns = ['c','paid','click','ctr','cpc'])
#output1.to_csv("G:/Business Analytics/Web Economics/code for we/non_lin_2_bidding.csv",index = False)


def non_lin_2(val,c,lamba = 5.2e-7): ####  best c = 8, click = 62 ####
    val.price = (( (val.pctr + np.sqrt( ((c**2) * (lamba**2) + val.pctr**2)))/(c * lamba))** (1/3) - ((c * lamba)/(val.pctr + np.sqrt(((c**2) * (lamba**2) + val.pctr**2)))) **1/3) * c
    val.got[val.price >= val.payprice] = 1
    val.got[val.price < val.payprice] = 0
    
    val1 = val.sample(frac = 1)
    val1 = np.array(val1)
    paid = 0
    click = 0
    impression = 0
    budget = 6250 * 1000
    for i in range(len(val1)):
        if paid + int(val1[i][3]) * val1[i][4] > budget:
            break
        else:
            paid = paid + int(val1[i][3]) * val1[i][4]
            click = click + val1[i][0] * val1[i][4]
            impression = impression + val1[i][4]
    output.append([c,paid, click, click/impression,paid/impression])
    return output

def non_lin_3():
    output = []
#
    for basebid in np.arange(1,100):
        val.price = basebid * ((val.pctr/avgctr) ** (4/2))
        val.price[val.price > 200] = 200
        val.price = val.price.round()
        val.got[val.price >= val.payprice] = 1
        val.got[val.price < val.payprice] = 0
        
        val1 = val.sample(frac = 1)
        val1 = np.array(val1)
        
        
        paid = 0
        click = 0
        impression = 0
        budget = 6250 * 1000
        for i in range(len(val1)):
            if paid + val1[i][3] * val1[i][4] > budget:
                break
            else:
                paid = paid + val1[i][3] * val1[i][4]
                click = click + val1[i][0] * val1[i][4]
                impression = impression + val1[i][4]
        output.append([basebid,paid, click, click/impression,paid/impression])
    
    
    output1 = pd.DataFrame(output,columns = ["basebid","paid","click", "CTR","CPC"])
    
    
    output1.to_csv("G:/Business Analytics/Web Economics/code for we/non_lin_4_bidding.csv")
    return output1

def non_lin_4():
    output = []
#
    for basebid in np.arange(1,100):
        val.price = basebid * ((val.pctr/avgctr) ** (5/2))
        val.price[val.price > 200] = 200
        val.price = val.price.round()
        val.got[val.price >= val.payprice] = 1
        val.got[val.price < val.payprice] = 0
        
        val1 = val.sample(frac = 1)
        val1 = np.array(val1)
        
        
        paid = 0
        click = 0
        impression = 0
        budget = 6250 * 1000
        for i in range(len(val1)):
            if paid + val1[i][3] * val1[i][4] > budget:
                break
            else:
                paid = paid + val1[i][3] * val1[i][4]
                click = click + val1[i][0] * val1[i][4]
                impression = impression + val1[i][4]
        output.append([basebid,paid, click, click/impression,paid/impression])
    
    
    output1 = pd.DataFrame(output,columns = ["basebid","paid","click", "CTR","CPC"])
    
    
    output1.to_csv("G:/Business Analytics/Web Economics/code for we/non_lin_4_bidding.csv")
    return output1


train = pd.read_csv("G:/Business Analytics/Web Economics/train.csv")
validation = pd.read_csv("G:/Business Analytics/Web Economics/validation.csv")
test = pd.read_csv("G:/Business Analytics/Web Economics/code for we/test.csv")

Train = encoding(train)
Validation = encoding(validation)
Test = encoding1(test)
Validation = Validation[Train.columns]
Test = Test[Train.columns.drop('click')]
Test.to_csv("G:/Business Analytics/Web Economics/code for we/test_encoding.csv",index = False)
Validation.to_csv("G:/Business Analytics/Web Economics/code for we/validation_encoding.csv",index = False)
Train.to_csv("G:/Business Analytics/Web Economics/code for we/train_encoding.csv",index = False)

a = pd.read_csv("G:/Business Analytics/Web Economics/code for we/train_encoding.csv")
b = pd.read_csv("G:/Business Analytics/Web Economics/code for we/test_encoding.csv")
#
trainX = a.drop("click",axis = 1)
trainY = a.click
trainX = np.array(trainX)
trainY = np.array(trainY)

valX = b

valX = np.array(valX)
valY = np.array(valY)

############logistic regression  ##############
clf = LogisticRegression(solver = 'sag')
clf = clf.fit(trainX,trainY)
clf = xgb.XGBClassifier()
clf= clf.fit(trainX,trainY)

valYY = clf.predict(valX)
valYP = clf.predict_proba(valX)

valYP1 = valYP[:,1]
valYP1 = recal(valYP1,0.025)
pctr = pd.DataFrame(valYP1,columns = ["pCTR"])
pctr.to_csv("G:/Business Analytics/Web Economics/code for we/pctr_test.csv",index = False)


validation = pd.read_csv("G:/Business Analytics/Web Economics/test1.csv")
A = validation[['bidid']]
lamba = 5.2e-7
c = 2.8
A['bidprice'] = 0
A.bidprice = np.sqrt(c * pctr.pCTR / lamba + c **2) - c
A = A.drop('price',axis = 1)
A.to_csv("G:/Business Analytics/Web Economics/code for we/bidprice_test.csv",index = False)
B = pd.read_csv("G:/Business Analytics/Web Economics/code for we/pctr_david.csv")
A.pctr = pctr
train = pd.read_csv("G:/Business Analytics/Web Economics/code for we/train.csv")
avgctr = train.click.sum()/len(train.click)
A.pctr = pctr
A['price'] = 0
val = A[['click','payprice','pctr','price']]
val['got'] = 0

output = []
#
for basebid in np.arange(1,100):
    val.price = basebid * ((val.pctr/avgctr)**(1))
    val.price[val.price > 200] = 200
    val.price = val.price.round()
    val.got[val.price >= val.payprice] = 1
    val.got[val.price < val.payprice] = 0
    
    val1 = val.sample(frac = 1)
    val1 = np.array(val1)
    
    
    paid = 0
    click = 0
    impression = 0
    budget = 6250 * 1000
    for i in range(len(val1)):
        if paid + val1[i][3] * val1[i][4] > budget:
            break
        else:
            paid = paid + val1[i][3] * val1[i][4]
            click = click + val1[i][0] * val1[i][4]
            impression = impression + val1[i][4]
    output.append([basebid,paid, click, click/impression,paid/impression])


output1 = pd.DataFrame(output,columns = ["basebid","paid","click", "CTR","CPC"])


####### Data Exploration  ###########
#a = []
#for i in train.columns:
#    a1 = len(train[i].unique())
#    a2 = len(train[i].unique())/len(validation[i].unique())
#    a.append([i,a1,a2])
#a = pd.DataFrame(a,columns = ['feature','num','rate'])
#week_ctr = []
#click = []
#for name in sorted(train.weekday.unique()):
#    week_ctr.append(len(train.loc[(train.click==1) & (train.weekday==name)])*100/len(train.loc[train.weekday==name]))
#    click.append(len(train.loc[(train.click==1) & (train.weekday==name)]))
#
#hour_ctr = []
#hour_click = []
#for name in sorted(train.hour.unique()):
#    hour_ctr.append(len(train.loc[(train.click==1) & (train.hour==name)])*100/len(train.loc[train.hour==name])) 
#    hour_click.append(len(train.loc[(train.click==1) & (train.hour==name)]))
#
#train["os"],train["browser"] = zip(*train.useragent.map(lambda x: x.split("_")))
#os_ctr = []
#os_click = []
#for name in train.os.unique():
#    os_ctr.append(len(train.loc[(train.click==1) & (train.os==name)])*100/len(train.loc[train.os==name]))
#    os_click.append(len(train.loc[(train.click==1) & (train.os==name)]))
#
#plt.bar(np.arange(len(train.os.unique())), os_click, tick_label=train.os.unique().tolist(), align='center')
#plt.ylabel('Click')
#plt.xlabel('Operating System')
#plt.show()