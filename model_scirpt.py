#-*- coding:utf-8 -*-

"""
	littlekid
	muyunlei@gmail.com
"""
import numpy as np
import scipy as sp
import scipy.sparse as sps
import util
import statdata
import feature
import smote
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


# ta=train_data.tolist()

# for i in xrange(len(ta)):
#     if ta[i][-1]==1:                                           
#         for j in xrange(113):
#             ta.append(ta[i])

# train_data=np.array(ta)

train_data = sp.genfromtxt('offline_train_beh_ui_feature0.csv',delimiter=',',skip_header=1)
val_data = sp.genfromtxt('offline_val_beh_ui_feature0.csv',delimiter=',',skip_header=1)

# feature_man_filter = []
# train_data = train_data[:,feature_man_filter]
# val_data = val_data[:,feature_man_filter]

feature_should_scale = []

# RandomForest
for i in xrange(2,train_data.shape[1]-1):
	train_data[:,i] = preprocessing.scale(train_data[:,i])
	val_data[:,i] = preprocessing.scale(val_data[:,i])

train_x = train_data[:,2:-2]
train_y = train_data[:,-1]
val_x = val_data[:,2:-2]
val_y = val_data[:,-1]


ne = 100
model = RandomForestClassifier(n_estimators = ne, n_jobs = 4, class_weight={0:1,1:3000})
model.fit(train_x,train_y)

pred = val_data[model.predict_proba(val_x)[:,1]>=0.95]

result = set()

for i in xrange(len(pred)):
	result.add((int(pred[i][0]),int(pred[i][1])))

val_set = set()

for e in util.parse('offline_val_label.csv'):
	val_set.add((int(e['user_id']),int(e['item_id'])))

util.offline_f1(result,val_set)

# user_geo,item_geo = util.ex_user_item_geohash('tianchi_mobile_recommend_train_user.csv','tianchi_mobile_recommend_train_item.csv')
user_item_non_geo = set()
for uik in result:
	user = str(uik[0])
	item = str(uik[1])
    if user_geo[user] != set() and item_geo[item] !=set():
    	if user_geo[user]&item_geo[item] == set():
            user_item_non_geo.add(uik)

for uik in user_item_non_geo:
	result.remove(uik)

util.offline_f1(result,val_set)

#################################################################################################
feature_used = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,\
				33,34,35,-2,-1]
meta_train_data = sp.genfromtxt('offline_train_beh_ui_feature3.csv',delimiter=',',skip_header=1)
meta_val_data = sp.genfromtxt('offline_val_beh_ui_feature3.csv',delimiter=',',skip_header=1)

# 412
meta_train_data = sp.genfromtxt('offline_val_sample_feature0.csv',delimiter=',',skip_header=1)
meta_val_data = sp.genfromtxt('offline_test_sample_feature0.csv',delimiter=',',skip_header=1)

# feature_should_scale = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,44,45,46]
# feature_should_scale = [2,3,4,5,8,11,19,20,21,22,24,27,28,29,30,39,40,43,44,48,49,50,51,53,54,\
						# 57,58,61,62,64,65,74,77,79,81]
# feature_used = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,\
# 				33,34,35,-2,-1]
# feature_should_scale = [2,3,4,5,8,11,19,20,21,22,24,27,28,29,30]

feature_should_scale = [2,3,4,5,8,11,19,20,21,22,24,27,28,29,30,37,40,43,44,47,48,49,\
						50,51,54,55,59,60,62,63,64,65,68,69,72,73,75,76,79,80,81,82,83,83,85,\
						86,89,91,93,94,98]


# LogisticRegression
for i in feature_should_scale:
	meta_train_data[:,i] = preprocessing.scale(meta_train_data[:,i])
	meta_val_data[:,i] = preprocessing.scale(meta_val_data[:,i])

fu1 = [0,1,2,3,4,5,6,7,-1]
fu2 = [0,1,2,3,4,5,6,7,8,9,10,15,16,17,18,34,19,20,21,22,23,24,25,26,35,27,28,29,30,31,32,33,34,35,68,69,70,71,72,73,-2,-1]
fu2 = [0,1,8,11,12,13,14,15,16,17,18,34,20,21,22,23,24,37,38,47,50,-1]
fu2 = [0,1,8,11,12,13,14,15,16,17,18,34,20,21,22,23,24,37,38,47,51,57,58,59,60,67,68,69,70,71,72,73,78,-1]

train_data = meta_train_data[:,fu2]
val_data = meta_val_data[:,fu2]

train_x = train_data[:,2:-1]
train_y = train_data[:,-1]
val_x = val_data[:,2:-1]
val_y = val_data[:,-1]

neg_train_data = train_data[train_y==0]
pos_train_data = train_data[train_y==1]

buy_data = train_data[train_data[:,29]>0]
val_buy_data = val_data[val_data[:,29]>0]

buy_x = buy_data[:,fu1]
buy_y = buy_data[:,-1]

val_buy_x = val_buy_data[:,fu1]


item_subset = set()
for e in util.parse('tianchi_mobile_recommend_train_item.csv'):
	item_subset.add(int(e['item_id']))

val_set = set()

for e in util.parse('offline_test_label.csv'):
	if int(e['item_id']) in item_subset:
		val_set.add((int(e['user_id']),int(e['item_id'])))

result_list = [set() for k in xrange(10)]
for i in xrange(10):
	rfc=RandomForestClassifier(n_estimators=50,n_jobs=4,class_weight={0:1,1:15},max_depth=5)
	rfc.fit(train_x,train_y)
	pred = val_data[rfc.predict_proba(val_x)[:,1]>=0.27]
	for i in xrange(len(pred)):
		result_list[i].add((int(pred[i][0]),int(pred[i][1])))


for p in np.arange(0.2,0.35,0.005):
	pred=val_data[rfc.predict_proba(val_x)[:,1]>=p]
	result=set()
	for i in xrange(len(pred)):
		result.add((int(pred[i][0]),int(pred[i][1])))
	print p
	user_item_non_geo = set()
	for uik in result:
		user = str(uik[0])
		item = str(uik[1])
		if user_geo[user] != set() and item_geo[item] !=set():
			if user_geo[user]&item_geo[item] == set():
				user_item_non_geo.add(uik)
	for uik in user_item_non_geo:
		result.remove(uik)
	util.offline_f1(result,val_set)



# feature selection pipeline
clf=Pipeline([('feature_selection',LogisticRegression(C=0.01,penalty='l1',dual=False)),('classification',RandomForestClassifier(n_estimators=50,class_weight={0:1,1:15},max_depth=5))])




neg_train_data = train_data[train_y==0]
pos_train_data = train_data[train_y==1]
pos_train_data_smote = smote.SMOTE(pos_train_data[:,2:],300,3)

# model training
# c = 1.0
proportion = 15
# n_models = neg_train_data.shape[0] / (pos_train_data.shape[0]*10)
n_models = 200

# model_list = [LogisticRegression(C=c,penalty='l2') for i in xrange(n_models)]
# pca_list = [PCA(n_components = 32) for i in xrange(n_models)]
model_list = [RandomForestClassifier(n_estimators = 100, n_jobs = -1, max_depth = 4) for i in xrange(n_models)]

for i in xrange(n_models):
	neg_sample_index = [np.random.randint(neg_train_data.shape[0]) for k in xrange(pos_train_data_smote.shape[0]*proportion)]
	neg_sample_data = neg_train_data[neg_sample_index]
	new_data = np.array(neg_sample_data[:,2:].tolist()+pos_train_data_smote.tolist())
	np.random.shuffle(new_data)
	new_x = new_data[:,:-1]
	new_y = new_data[:,-1]
	model_list[i].fit(new_x,new_y)
	print "model#%d Train Score: %f"%(i+1,model_list[i].score(new_x,new_y))

# result generate

prob_threshold = 0.6

result_vote = np.zeros(val_data.shape[0])

for i in xrange(n_models):
	result_vote += np.array(model_list[i].predict_proba(val_x)[:,1]>=prob_threshold,dtype=int)

vote_prop = 0.72
pred = val_data[result_vote >= (n_models*vote_prop)]

result = set()

for i in xrange(len(pred)):
	result.add((str(int(pred[i][0])),str(int(pred[i][1]))))

util.offline_f1(result,item_subset,val_set)


# user_geo,item_geo = util.ex_user_item_geohash('tianchi_mobile_recommend_train_user.csv','tianchi_mobile_recommend_train_item.csv')
user_item_non_geo = set()
for uik in result:
	user = str(uik[0])
	item = str(uik[1])
    if user_geo[user] != set() and item_geo[item] !=set():
    	if user_geo[user]&item_geo[item] == set():
            user_item_non_geo.add(uik)

for uik in user_item_non_geo:
	result.remove(uik)

util.offline_f1(result,item_subset,val_set)






#################################################################################################
# online predict 
train_data = sp.genfromtxt('offline_val_beh_ui_feature3.csv',delimiter=',',skip_header=1)
test_data = sp.genfromtxt('online_beh_ui_feature3.csv',delimiter=',',skip_header=1)

# feature_should_scale = [2,3,4,5,8,11,19,20,21,22,24,27,28,29,30]
feature_should_scale = [2,3,4,5,8,11,19,20,21,22,24,27,28,29,30,34,35,36,37,40,43,51,\
						52,53,54,56,59,60,61,62,66,67,68,69,72,75,82,83,84,85,87,90,91,92,93]
for i in feature_should_scale:
	train_data[:,i] = preprocessing.scale(train_data[:,i])
	test_data[:,i] = preprocessing.scale(test_data[:,i])

train_x = train_data[:,2:-1]
train_y = train_data[:,-1]
test_x = test_data[:,2:-1]

neg_train_data = train_data[train_y==0]
pos_train_data = train_data[train_y==1]

c = 2.0
proportion = 15


# n_models = neg_train_data.shape[0] / (pos_train_data.shape[0]*10)
n_models = 300

model_list = [LogisticRegression(C=c) for i in xrange(n_models)]

for i in xrange(n_models):
	neg_sample_index = [np.random.randint(neg_train_data.shape[0]) for k in xrange(pos_train_data.shape[0]*proportion)]
	neg_sample_data = neg_train_data[neg_sample_index]
	new_data = np.array(neg_sample_data.tolist()+pos_train_data.tolist())
	np.random.shuffle(new_data)
	new_x = new_data[:,2:-1]
	new_y = new_data[:,-1]
	model_list[i].fit(new_x,new_y)
	print "Model#%d:Train Score: %f"%(i,model_list[i].score(new_x,new_y))

prob_threshold = 0.78
vote_prop = 0.96
result_vote = np.zeros(test_data.shape[0])
for i in xrange(n_models):
	result_vote += np.array(model_list[i].predict_proba(test_x)[:,1]>=prob_threshold,dtype=int)

pred = test_data[result_vote >= (n_models*vote_prop)]
result = set()

for i in xrange(len(pred)):
	result.add((int(pred[i][0]),int(pred[i][1])))

print len(result),

user_item_non_geo = set()

for uik in result:
	user = str(uik[0])
	item = str(uik[1])
    if user_geo[user] != set() and item_geo[item] !=set():
    	if user_geo[user]&item_geo[item] == set():
            user_item_non_geo.add(uik)

for uik in user_item_non_geo:
	result.remove(uik)

print len(result)

util.output_pred_result(result,'online_predict_behf_ui_bagging_lr_407.csv')


# generate online predition 408
# LR + RF
# LR: C=0.1 class_weight = {0:1,1:15} prob_threshold = 0.22
# RF: 50T class_weight = {0:1,1:15} max_depth = 5 prob_threshold=0.25
meta_train_data = sp.genfromtxt('offline_val_beh_ui_feature3.csv',delimiter=',',skip_header=1)
meta_test_data = sp.genfromtxt('online_test_beh_ui_feature3.csv',delimiter=',',skip_header=1)

# user_geo,item_geo = util.ex_user_item_geohash('tianchi_mobile_recommend_train_user.csv','tianchi_mobile_recommend_train_item.csv')

feature_should_scale = [2,3,4,5,8,11,19,20,21,22,24,27,28,29,30,39,40,43,44,48,49,50,51,53,54,\
						57,58,61,62,64,65,74,77,79]
# feature_used = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,\
# 				33,-2,-1]


# fu2 = [0,1,2,3,4,5,6,7,8,9,10,15,16,17,18,34,19,20,21,22,23,24,25,26,35,27,28,29,30,31,32,33,34,35,68,69,70,71,72,73,-2,-1]

fu2 = [0,1,8,9,10,11,12,13,14,15,16,17,18,34,19,20,21,22,23,24,25,26,35,27,28,29,30,31,32,33,45,46,79,80,81,82,83,84,-1]
for i in feature_should_scale:
	meta_train_data[:,i] = preprocessing.scale(meta_train_data[:,i])
	meta_test_data[:,i] = preprocessing.scale(meta_test_data[:,i])


train_data = meta_train_data[:,fu2]
test_data = meta_test_data[:,fu2]

train_x = train_data[:,2:-1]
train_y = train_data[:,-1]
test_x = test_data[:,2:-1]


rfc = RandomForestClassifier(n_estimators=100,n_jobs=4,class_weight={0:1,1:15},max_depth=5)

rfc.fit(train_x,train_y)

prob_threshold = 0.93

pred = test_data[clf.predict_proba(test_x)[:,1]>=prob_threshold]

result = set()

for i in xrange(len(pred)):
	result.add((int(pred[i][0]),int(pred[i][1])))

print len(result),

user_item_non_geo = set()

for uik in result:
	user = str(uik[0])
	item = str(uik[1])
    if user_geo[user] != set() and item_geo[item] !=set():
    	if user_geo[user]&item_geo[item] == set():
            user_item_non_geo.add(uik)

for uik in user_item_non_geo:
	result.remove(uik)

print len(result)

util.output_pred_result(result,'online_predict_behf_ui_RF_411.csv')



# 413
# sample feature
m = 10
for i in xrange(m):
	feature_file = 'online_sample/online_train_sample_feature0'+str(i)+'.csv'
	feature.ex_behav_user_item_feature_md('offline_test.csv','offline_test_label.csv',\
										'tianchi_mobile_recommend_train_item.csv',\
										feature_file,datetime(2014,12,18,0,0))

# test feature

feature.ex_behav_user_item_test_feature('offline_test.csv',\
										'tianchi_mobile_recommend_train_item.csv',\
										'offline_test_feature413.csv',datetime(2014,12,18,0,0))

# read data to ram
# train #m RandomForest Classifier
item_subset = set()
for e in util.parse('tianchi_mobile_recommend_train_item.csv'):
	item_subset.add(int(e['item_id']))

val_set = set()
for e in util.parse('offline_test_label.csv'):
	if int(e['item_id']) in item_subset:
		val_set.add((int(e['user_id']),int(e['item_id'])))

fu_test = [0,1,6,7,8,9,10,39,40,11,12,13,14,15,16,17,18,34,19,20,21,22,23,24,25,26,43,44,32,33,82,83,84]
fu_train = [0,1,6,7,8,9,10,39,40,11,12,13,14,15,16,17,18,34,19,20,21,22,23,24,25,26,43,44,32,33,82,83,84,-1]

fu_test = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,39,40,41,42,43,44,45,46,47,48,49]
fu_train = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,39,40,41,42,43,44,45,46,47,48,49,-1]

fu_test = [0,1,6,7,8,11,19,20,21,22,23,24,25,26,43,44,32,33,45,46,75,76,77,78,82,83,84]
fu_train = [0,1,6,7,8,11,19,20,21,22,23,24,25,26,43,44,32,33,45,46,75,76,77,78,82,83,84,-1]


meta_test_data = sp.genfromtxt('offline_test_feature413.csv',delimiter=',',skip_header=1,usecols=fu_test)
test_x = meta_test_data[:,2:]
prob_threshold = 0.50


# feature_file_list = ['offline_sample/offline_val_sample_feature0'+str(i)+'.csv' for i in xrange(m)]
feature_file_list = ['online_sample/online_train_sample_feature0'+str(i)+'.csv' for i in xrange(m)]

model_list = [RandomForestClassifier(n_estimators = 100, n_jobs = -1,class_weight={1:45},max_depth=5) for i in xrange(m)]
result = [set() for i in xrange(m)]

for i in xrange(test_x.shape[1]):
	test_x[:,i] = preprocessing.scale(test_x[:,i])


for k in xrange(m):
	meta_train_data = sp.genfromtxt(feature_file_list[k],delimiter=',',skip_header=1,usecols=fu_train)
	train_x = meta_train_data[:,2:-1]
	for i in xrange(train_x.shape[1]):
		train_x[:,i] = preprocessing.scale(train_x[:,i])
	train_y = meta_train_data[:,-1]
	model_list[k].fit(train_x,train_y)
	pred = meta_test_data[model_list[k].predict_proba(test_x)[:,1]>=prob_threshold]
	for i in xrange(len(pred)):
		result[k].add((int(pred[i][0]),int(pred[i][1])))
	print "#%d model f1:"%(k)
	util.offline_f1(result[k],item_subset,val_set)


vote_prop = 1
result_vote = np.zeros(meta_test_data.shape[0])

for i in xrange(m):
	result_vote += np.array(model_list[i].predict_proba(test_x)[:,1]>=prob_threshold,dtype=int)

pred = meta_test_data[result_vote >= (m*vote_prop)]

vote_result = set()
for i in xrange(len(pred)):
	vote_result.add((int(pred[i][0]),int(pred[i][1])))

util.offline_f1(vote_result,item_subset,val_set)



pred15 = val_data[rfc15.predict_proba(val_x15)[:,1]>=0.1]
pred16 = val_data[rfc16.predict_proba(val_x16)[:,1]>=0.15]
result15=set()
for i in xrange(len(pred15)):
	result15.add((str(int(pred15[i][0])),str(int(pred15[i][1]))))

result16=set()
for i in xrange(len(pred16)):
	result16.add((str(int(pred16[i][0])),str(int(pred16[i][1]))))

print len(result15),len(result16)
util.offline_f1(result15&result16,item_subset,val_set)

