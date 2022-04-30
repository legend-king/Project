from django.shortcuts import render,HttpResponse
from django.http import QueryDict
import pandas as pd

df = pd.read_csv('E:\\phishing.csv')
dataset = df.values
X = dataset[:,1:19]
Y = dataset[:,20]
from sklearn import preprocessing
 
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
 
# Encode labels in column 'class'.
df['class']= label_encoder.fit_transform(df['class'])
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size=0.3,random_state = 40)
from keras.models import Sequential
from sklearn import metrics
from keras.layers import Dense, Embedding, SimpleRNN, GRU, LSTM, SpatialDropout1D
max_fatures = 2000
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(SimpleRNN(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
# print(model.summary())
model.compile(loss='mse',optimizer='Rmsprop',metrics=['accuracy'])
hist = model.fit(X_train, Y_train,  batch_size=200, epochs=10,validation_data=(X_test,Y_test))
model.evaluate(X_test, Y_test)[1]
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score)
from keras.preprocessing import sequence
from keras.models import Sequential
from time import time
t1=time()
y_pred = model.predict(X_test)
# print("predict time:", round(time()-t1, 3), "s")
accuracy = accuracy_score(Y_test, y_pred.round(), normalize=True)
recall = recall_score(Y_test, y_pred.round() , average="weighted")
precision = precision_score(Y_test, y_pred.round() , average="weighted")
f1=  f1_score(Y_test, y_pred.round(), average="weighted")
# print("")
# print("accuracy")
# print("%.6f" %accuracy)
# print("")
# print("racall")
# print("%.6f" %recall)
# print("")
# print("precision")
# print("%.6f" %precision)
# print("")
# print("f1score")
# print("%.6f" %f1)

# Create your views here.
def index(request):
    if request.method=="POST":
        d = dict()
        lst=[]
        d['Index'] = int(request.POST.get('Index'))
        d['UsingIP'] = int(request.POST.get('UsingIP'))
        d['LongURL'] = int(request.POST.get('LongURL'))
        d['ShortURL'] = int(request.POST.get('ShortURL'))
        d['Symbol@'] = int(request.POST.get('Symbol@'))
        d['Redirecting//'] = int(request.POST.get('Redirecting//'))
        d['PrefixSuffix-']=int(request.POST.get('PrefixSuffix-'))
        d['SubDomains']=int(request.POST.get('SubDomains'))
        d['HTTPS']=int(request.POST.get('HTTPS'))
        d['DomainRegLen'] = int(request.POST.get('DomainRegLen'))
        d['Favicon'] = int(request.POST.get('Favicon'))
        d['NonStdPort'] = int(request.POST.get('NonStdPort'))
        d['HTTPSDomainURL'] = int(request.POST.get('HTTPSDomainURL'))
        d['RequestURL'] = int(request.POST.get('RequestURL'))
        d['AnchorURL'] = int(request.POST.get('AnchorURL'))
        d['LinksInScriptTags']=int(request.POST.get('LinksInScriptTags'))
        d['ServerFormHandler']=int(request.POST.get('ServerFormHandler'))
        d['InfoEmail']=int(request.POST.get('InfoEmail'))
        d['AbnormalURL'] = int(request.POST.get('AbnormalURL'))
        d['WebsiteForwarding'] = int(request.POST.get('WebsiteForwarding'))
        d['StatusBarCust'] = int(request.POST.get('StatusBarCust'))
        d['DisableRightClick'] = int(request.POST.get('DisableRightClick'))
        d['UsingPopupWindow'] = int(request.POST.get('UsingPopupWindow'))
        d['IframeRedirection'] = int(request.POST.get('IframeRedirection'))
        d['AgeofDomain']=int(request.POST.get('AgeofDomain'))
        d['DNSRecording']=int(request.POST.get('DNSRecording'))
        d['WebsiteTraffic']=int(request.POST.get('WebsiteTraffic'))
        d['PageRank'] = int(request.POST.get('PageRank'))
        d['GoogleIndex'] = int(request.POST.get('GoogleIndex'))
        d['LinksPointingToPage']=int(request.POST.get('LinksPointingToPage'))
        d['StatsReport']=int(request.POST.get('StatsReport'))
        for i in d.values():
            lst.append(i)
        lst1 = [lst]
        y=model.predict(lst1)
        print("prediction ",y)
        context={'data':accuracy}
        print(d)
        return render(request,'index.html',context)
    context={'data':""}
    return render(request, 'index.html',context)