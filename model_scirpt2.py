#-*- coding:utf-8 -*-


import numpy as np
import scipy as sp
import util
import feature
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from datetime import *

item_subset = set()
for e in util.parse('tianchi_mobile_recommend_train_item.csv'):
    item_subset.add(e['item_id'])


# oneday feature offline test
train_file = 'oneday_feature1217.csv'
val_file = 'oneday_feature1218.csv'

train_data = sp.genfromtxt(train_file,delimiter=',',skip_header=1)
val_data = sp.genfromtxt(val_file,delimiter=',',skip_header=1)

fs = [2,3,4,5]

for i in fs:
	train_data[:,i] = preprocessing.scale(train_data[:,i])
	val_data[:,i] = preprocessing.scale(val_data[:,i])

train_x = train_data[:,2:-1]
train_y = train_data[:,-1]

val_x = val_data[:,2:-1]

rfc = RandomForestClassifier(n_estimators=600,n_jobs=-1,class_weight={1:15},max_depth=4,random_state=None)

rfc.fit(train_x,train_y)

val_set = set()
for e in util.parse('offline_label_1218.csv'):
	val_set.add((e['user_id'],e['item_id']))

test_set = set()
for e in util.parse('offline_label_1218.csv'):
	test_set.add((e['user_id'],e['item_id']))


for p in np.arange(0.25,0.45,0.01):
	pred=val_data[rfc.predict_proba(val_x)[:,1]>=p]
	result=set()
	for i in xrange(len(pred)):
		result.add((str(int(pred[i][0])),str(int(pred[i][1]))))
	print p
	f1 = util.offline_f1(result,item_subset,val_set)

for p in np.arange(0.1,0.45,0.01):
	pred=val_data[rfc.predict_proba(val_x)[:,1]>=p]
	print p,len(pred)


# validation
# train a model
train_file = 'oneday_feature1217.csv'

train_data = sp.genfromtxt(train_file,delimiter=',',skip_header=1)

fs = [2,3,4,5,6,7,8,9]

for i in fs:
	train_data[:,i] = preprocessing.scale(train_data[:,i])

rfc = RandomForestClassifier(n_estimators=600,n_jobs=-1,class_weight={1:15},max_depth=4,random_state=9)

rfc.fit(train_x,train_y)

best_f1 = [0 for i in xrange(1,18)]
best_p = [0 for i in xrange(1,18)]
for i in xrange(1,18):
	val_file = 'oneday_feature12' + str(i) +'.csv' if i/10!=0 else 'oneday_feature12' + '0' +str(i) +'.csv'
	val_label_file = 'offline_label_12' + str(i+1) +'.csv' if (i+1)/10!=0 else 'offline_label_12' + '0' +str(i+1) +'.csv'
	print val_file,val_label_file
	if val_file == train_file:
		continue
	val_set = set()
	for e in util.parse(val_label_file):
		val_set.add((e['user_id'],e['item_id']))
	val_data = sp.genfromtxt(val_file,delimiter=',',skip_header=1)
	for k in fs:
		val_data[:,k] = preprocessing.scale(val_data[:,k])
	val_x = val_data[:,2:-2] 
	for p in np.arange(0.1,0.45,0.01):
		pred=val_data[rfc.predict_proba(val_x)[:,1]>=p]
		result=set()
		for k in xrange(len(pred)):
			result.add((str(int(pred[k][0])),str(int(pred[k][1]))))
		f1 = util.offline_f1(result,item_subset,val_set)
		if f1 > best_f1[i-1]:
			best_f1[i-1] = f1
			best_p[i-1] = p

# everday ensemble
val_file = 'oneday_feature1217.csv'

val_data = sp.genfromtxt(val_file,delimiter=',',skip_header=1)

fs = [2,3,4,5,6,7,8,9]

for i in fs:
	val_data[:,i] = preprocessing.scale(val_data[:,i])

val_x = val_data[:,2:-2]

val_set = set()
for e in util.parse('offline_label_1218.csv'):
	val_set.add((e['user_id'],e['item_id']))

result = set()
best_f1 = [0 for i in xrange(1,17)]
best_p = [0 for i in xrange(1,17)]
for i in xrange(1,17):
	train_file = 'oneday_feature12' + str(i) +'.csv' if i/10!=0 else 'oneday_feature12' + '0' +str(i) +'.csv'
	train_data = sp.genfromtxt(train_file,delimiter=',',skip_header=1)
	print train_file
	for k in fs:
		train_data[:,k] = preprocessing.scale(train_data[:,k])
	train_x = train_data[:,2:-2]
	train_y = train_data[:,-1]
	rfc = RandomForestClassifier(n_estimators=600,n_jobs=-1,class_weight={1:15},max_depth=4,random_state=None)
	rfc.fit(train_x,train_y)
	for p in np.arange(0.1,0.45,0.01):
		pred=val_data[rfc.predict_proba(val_x)[:,1]>=p]
		result_s = set()
		for k in xrange(len(pred)):
			result_s.add((str(int(pred[k][0])),str(int(pred[k][1]))))
		f1 = util.offline_f1(result_s,item_subset,val_set)
		if f1 > best_f1[i-1]:
			best_f1[i-1] = f1
			best_p[i-1] = p
	pred=val_data[rfc.predict_proba(val_x)[:,1]>=best_p[i-1]]
	result_s = set()
	for k in xrange(len(pred)):
		result_s.add((str(int(pred[k][0])),str(int(pred[k][1]))))
	util.offline_f1(result_s,item_subset,val_set)
	if i == 1:
		result = result_s
	else:
		result = result&result_s
	util.offline_f1(result,item_subset,val_set)

util.offline_f1(result,item_subset,val_set)



# item category stat
category_count = dict()
sday = datetime(2014,12,18)
for e in util.parse('tianchi_mobile_recommend_train_user.csv'):
	dat = util.time_proc(e['time'])
	user = e['user_id']
	item = e['item_id']
	cate = e['item_category']
	if e['behavior_type'] == '4' and item in item_subset and dat.date() <= sday.date():
		if cate not in category_count:
			category_count[cate] = 0
		category_count[cate] += 1

item_category = dict()
for e in util.parse('tianchi_mobile_recommend_train_item.csv'):
	item_category[e['item_id']] = e['item_category']

la = set()
item_ab_cate = set()
for c in category_count:
	if category_count[c] <= 1:
		item_ab_cate.add(c)

item_ab = set()
for it in item_category:
	if item_category[it] in item_ab_cate:
		item_ab.add(it)

for uik in result:
	if uik[1] in item_ab:
		la.add(uik)


user_count = dict()
for e in util.parse('tianchi_mobile_recommend_train_user.csv'):
	dat = util.time_proc(e['time'])
	user = e['user_id']
	item = e['item_id']
	if item in item_subset and dat.date() <= sday.date():
		if user not in user_count:
			user_count[user] = 0
		user_count[user] += 1


lc = set()
ld = set()
ui_set_o3 = set()
ui_set_a3 = set()
day2 = set([(sday-timedelta(1)).date(),(sday-timedelta(2)).date()])

day3 = set([sday.date(),(sday-timedelta(1)).date(),(sday-timedelta(2)).date()])

for e in util.parse('tianchi_mobile_recommend_train_user.csv'):
	dat = util.time_proc(e['time'])
	user = e['user_id']
	item = e['item_id']
	if e['behavior_type'] == '4' and item in item_subset and dat.date() in day2:
		lc.add((user,item))

for e in util.parse('tianchi_mobile_recommend_train_user.csv'):
	dat = util.time_proc(e['time'])
	user = e['user_id']
	item = e['item_id']
	if item in item_subset and dat.date() in day3:
		ui_set_a3.add((user,item))
	if item in item_subset:
		ui_set_o3.add((user,item))

ui_count = dict()
for e in util.parse('tianchi_mobile_recommend_train_user.csv'):
	dat = util.time_proc(e['time'])
	user = e['user_id']
	item = e['item_id']
	uik = (user,item)
	beh = int(e['behavior_type'])-1
	if item in item_subset and dat.date() in day3:
		if uik not in ui_count:
			ui_count[uik] = [0 for i in xrange(4)]
		ui_count[uik][beh] += 1
