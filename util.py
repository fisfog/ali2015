#-*- coding:utf-8 -*-

"""
	littlekid
	muyunlei@gmail.com
"""
# import gzip
import numpy as np
from datetime import *


def parse(csvfile):
	"""
	解析数据
	"""
	f = open(csvfile, 'r')
	head = f.readline().strip('\r\n').split(',')
	n = len(head)
	entry = {}
	for l in f:
		l = l.strip('\r\n').split(',')
		for i in xrange(n):
			entry[head[i]] = l[i]
		yield entry
	

def time_proc(time_str):
	"""
	将时间字符串转化为python datetime对象
	"""
	time = {}
	part = time_str.split()
	time['h'] = int(part[-1])
	ymd = part[0].split('-')
	time['y'] = int(ymd[0])
	time['m'] = int(ymd[1])
	time['d'] = int(ymd[2])
	return datetime(time['y'],time['m'],time['d'],time['h'])

# 411之前划分数据集函数
def split_offline_train_val(totalfile,item_table,train_name,train_label,val_name,val_label):
	# user_time = {}
	# def tran(tm):
	# 	return tm['y']*1000000+tm['m']*10000+tm['d']*100+tm['h']
	# for e in parse(totalfile):
	# 	if e['user_id'] not in user_time:
	# 		user_time[e['user_id']] = []
	# 	t = time_proc(e['time'])
	# 	user_time[e['user_id']].append(tran(t))
	item_in_sub = set()
	for e in parse(item_table):
		item_in_sub.add(e['item_id'])
	f_train = open(train_name,'w')
	f_label = open(train_label,'w')
	f_val = open(val_label,'w')
	f_val_set = open(val_name,'w')
	f_train.write('user_id,item_id,behavior_type,user_geohash,item_category,time\n')
	f_val_set.write('user_id,item_id,behavior_type,user_geohash,item_category,time\n')
	f_label.write('user_id,item_id\n')
	f_val.write('user_id,item_id\n')
	for e in parse(totalfile):
		# if tran(time_proc(e['time'])) != max(user_time[e['user_id']]):
		# 	f_train.write(e['user_id']+','+e['item_id']+','+e['behavior_type']+','\
		# 				+e['user_geohash']+','+e['item_category']+','+e['time']+'\n')
		# else:
		# 	f_val.write(e['user_id']+','+e['item_id']+'\n')
		t = time_proc(e['time'])
		if t.month==12 and t.day==18 and e['behavior_type']=='4' and e['item_id'] in item_in_sub:
			f_val.write(e['user_id']+','+e['item_id']+'\n')
		if t.month==12 and t.day==17 and e['behavior_type']=='4' and e['item_id'] in item_in_sub:
			f_label.write(e['user_id']+','+e['item_id']+'\n')
		if not (t.month==12 and t.day==18):
			f_val_set.write(e['user_id']+','+e['item_id']+','+e['behavior_type']+','\
						+e['user_geohash']+','+e['item_category']+','+e['time']+'\n')
		if not (t.month==12 and (t.day==18 or t.day==17)):
			f_train.write(e['user_id']+','+e['item_id']+','+e['behavior_type']+','\
						+e['user_geohash']+','+e['item_category']+','+e['time']+'\n')
	f_train.close()
	f_label.close()
	f_val_set.close()
	f_val.close()


# 411 new function
def split_dataset(totalfile,split_date,outfile,out_lable):
	# extract item subset
	# item_in_sub = set()
	# for e in parse(item_subset_file):
	# 	item_in_sub.add(e['item_id'])
	# split dataset with the split_date
	f_out = open(outfile,'w')
	f_label = open(out_lable,'w')
	f_out.write('user_id,item_id,behavior_type,user_geohash,item_category,time\n')
	f_label.write('user_id,item_id\n')
	for e in parse(totalfile):
		t = time_proc(e['time'])
		if t.month == split_date.month and t.day == split_date.day and e['behavior_type']=='4':
			f_label.write(e['user_id']+','+e['item_id']+'\n')
		if (t.month == split_date.month and t.day < split_date.day) or t.month < split_date.month:
			f_out.write(e['user_id']+','+e['item_id']+','+e['behavior_type']+','\
						+e['user_geohash']+','+e['item_category']+','+e['time']+'\n')

	f_out.close()
	f_label.close()



def z_score(x,mu,sigma):
	return (x-mu)/sigma


def avg_time_diff(time_set):
	if len(time_set) <= 1:
		return 0

	time_array = np.array(list(time_set))
	time_array.sort()
	s = 0
	for i in xrange(1,len(time_array)):
		A = time_array[i]
		B = time_array[i-1]
		s += (A-B).total_seconds()/3600
	return s/(len(time_array)-1)

def max_time_diff(time_set):
	if len(time_set) <= 1:
		return 0
	time_array = np.array(list(time_set))
	time_array.sort()
	max_t_diff = -1
	s = 0
	for i in xrange(1,len(time_array)):
		A = time_array[i]
		B = time_array[i-1]
		s = (A-B).total_seconds()/3600
		if s > max_t_diff:
			max_t_diff = s
	return max_t_diff

def min_time_diff(time_set):
	if len(time_set) <= 1:
		return 0
	time_array = np.array(list(time_set))
	time_array.sort()
	min_t_diff = 31
	s = 0
	for i in xrange(1,len(time_array)):
		A = time_array[i]
		B = time_array[i-1]
		s = (A-B).total_seconds()/3600
		if s < min_t_diff:
			min_t_diff = s
	return min_t_diff

def avg_click_diff_hour(click_time_set, buy_time_set):
	if len(buy_time_set) == 0:
		return 0
	avg_h = 0
	i = 0
	for d in np.sort(list(buy_time_set)):
		click_date = np.sort(list(click_time_set))
		while i < len(click_date) and click_date[i] <= d:
			i += 1
		avg_h += (d - click_date[i]).total_seconds() / 3600
	return avg_h*1.0/len(buy_time_set)


def output_pred_result(result,outfile):
	f=open(outfile,'w')
	f.write('user_id,item_id\n')
	for (user,item) in result:
		f.write(str(user)+','+str(item)+'\n')
	f.close()

def max_proba(estimator,item_subset,val_set,a,b,data,x,):
	max_f1 = 0
	max_p = a
	for p in np.arange(a,b,0.01):
		pred = data[estimator.predict_proba(x)[:,1] >= p]
		result=set()
		for i in xrange(len(pred)):
			result.add((str(int(pred[i][0])),str(int(pred[i][1]))))
		print p,
		f1 = offline_f1(result,item_subset,val_set)
		if f1 > max_f1:
			max_f1 = f1
			max_p = p
	return max_f1,max_p

def ex_user_item_geohash(user_file,item_subset_file,d):
	user_geo = {}
	item_geo = {}
	for e in parse(user_file):
		user = e['user_id']
		if user not in user_geo:
			user_geo[user] = set()
		if e['user_geohash'] != '':
			user_geo[user].add(e['user_geohash'][:d])

	for e in parse(item_subset_file):
		item = e['item_id']
		if item not in item_geo:
			item_geo[item] = set()
		if e['item_geohash'] != '':
			item_geo[item].add(e['item_geohash'][:d])
	return user_geo,item_geo

def offline_f1(pred_set,item_subset,val_set):
	pred_sub_set = set()
	for uik in pred_set:
		if uik[1] in item_subset:
			pred_sub_set.add(uik)
	hit = len(pred_sub_set&val_set)
	if hit == 0:
		return 0
	pred_size = len(pred_sub_set)
	val_size = len(val_set)
	prec = hit*1.0/pred_size
	rec = hit*1.0/val_size
	f1 = 2.0*prec*rec/(prec+rec)
	print "#hit:%d in %d, Precision:%f Recall:%f F1:%f"%(hit,pred_size,prec,rec,f1)
	return f1

def cal_f1(predict_table,val_table):
	pred_list = set()
	val_list = set()
	for e in parse(predict_table):
		pred_list.add((e['user_id'],e['item_id']))
	for e in parse(val_table):
		val_list.add((e['user_id'],e['item_id']))

	hit = 0
	pred_size = len(pred_list)
	val_size = len(val_list)
	for p in pred_list:
		if p in val_list:
			hit += 1
	prec = hit*1.0/pred_size
	rec = hit*1.0/val_size
	if hit==0:
		print 0
		return 0
	f1 = 2.0*prec*rec/(prec+rec)
	print "#hit:%d Precision:%f Recall:%f F1:%f"%(hit,prec,rec,f1)
	return f1

