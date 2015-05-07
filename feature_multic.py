#-*- coding:utf-8 -*-

"""
	littlekid
	muyunlei@gmail.com
"""

import thread
import numpy as np
import scipy as sp
import scipy.sparse as sps
import util
import statdata
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from datetime import *



def ex_meta_feature(infile,item_subset,sday,out_feature):
	ui_click_feature = dict()
	ui_collect_feature = dict()
	ui_cart_feature = dict()
	ui_buy_feature = dict()

	ui_click_stat = dict()
	ui_collect_stat = dict()
	ui_cart_stat = dict()
	ui_buy_stat = dict()

	item_category = dict()
	ui_set = set()
	ui_label = set()
	label_day = sday + timedelta(1)
	time_span_a = sday - timedelta(5)
	start_day = datetime(2014,11,18)

	def ex_uif():
		n_click_feature = 9
		n_collect_feature = 5
		n_cart_feature = 8
		n_buy_featuure = 6
		print "extract ui feature"
		for e in util.parse(infile):
			user = e['user_id']
			item = e['item_id']
			date_time = util.time_proc(e['time'])
			uik = (user,item)
			beh = int(e['behavior_type'])
			if item not in item_subset:
				continue
			item_category[item] = e['item_category']
			if date_time.date() == label_day.date() and beh == 4:
				ui_label.add(uik)
			if date_time.date() < label_day.date():
				if uik not in ui_set:
					ui_click_feature[uik] = [0 for i in xrange(n_click_feature)]
					ui_click_stat[uik] = [set() for i in xrange(1)]
					ui_collect_feature[uik] = [0 for i in xrange(n_collect_feature)]
					ui_collect_stat[uik] = [set() for i in xrange(1)]
					ui_cart_feature[uik] = [0 for i in xrange(n_cart_feature)]
					ui_cart_stat[uik] = [set() for i in xrange(1)]
					ui_buy_feature[uik] = [0 for i in xrange(n_buy_featuure)]
					ui_buy_stat[uik] = [set() for i in xrange(1)]
					ui_set.add(uik)
		
				if beh == 1:
					ui_click_feature[uik][0] += 1
					ui_click_stat[uik][0].add(date_time)
					if date_time.date() == sday.date():
						ui_click_feature[uik][3] += 1
					if date_time.date() >= time_span_a.date() and date_time.date() < sday.date():
						ui_click_feature[uik][6] += 1
		
				if beh == 2:
					ui_collect_feature[uik][0] += 1
					ui_collect_stat[uik][0].add(date_time)
					if date_time.date() == sday.date():
						ui_collect_feature[uik][1] = 1
					if date_time.date() >= time_span_a.date() and date_time.date() < sday.date():
						ui_collect_feature[uik][4] = 1
		
				if beh == 3:
					ui_cart_feature[uik][0] += 1
					ui_cart_stat[uik][0].add(date_time)
					if date_time.date() == sday.date():
						ui_cart_feature[uik][2] += 1
					if date_time.date() >= time_span_a.date() and date_time.date() < sday.date():
						ui_cart_feature[uik][5] += 1
		
				if beh == 4:
					ui_buy_feature[uik][0] += 1
					ui_buy_stat[uik][0].add(date_time)
					if date_time.date() == sday.date():
						ui_buy_feature[uik][2] += 1
					if date_time.date() >= time_span_a.date() and date_time.date() < sday.date():
						ui_buy_feature[uik][4] += 1
	
	
		print "Total <u,i> pair:%d"%(len(ui_set))
	
		for uik in ui_set:
			click_day = set([d.date() for d in ui_click_stat[uik][0]])
			cart_day = set([d.date() for d in ui_cart_stat[uik][0]])
			buy_day = set([d.date() for d in ui_buy_stat[uik][0]])
			click_day_span = set([d for d in click_day if d >= time_span_a.date() and d < sday.date()])
			cart_day_span = set([d for d in cart_day if d >= time_span_a.date() and d < sday.date()])
			buy_day_span = set([d for d in buy_day if d >= time_span_a.date() and d < sday.date()])
	
	
			# click
			ui_click_feature[uik][1] = len(click_day)
			if ui_click_feature[uik][1] != 0:
				ui_click_feature[uik][2] = 1.0*ui_click_feature[uik][0]/ui_click_feature[uik][1]
			if len(ui_click_stat[uik][0]) != 0:
				last_click_time = np.sort(np.array(list(ui_click_stat[uik][0])))[-1]
				ui_click_feature[uik][4] = 3600/(label_day-last_click_time).total_seconds()
				if len(ui_buy_stat[uik][0]) != 0:
					last_buy_time = np.sort(np.array(list(ui_buy_stat[uik][0])))[-1]
					if last_click_time.date() == sday.date() and last_buy_time >= last_click_time:
						ui_click_feature[uik][7] = 1
			ui_click_feature[uik][8] = len(click_day_span)
			ui_click_feature[uik][5] = len(click_day&buy_day)
	
			# collect
			if len(ui_collect_stat[uik][0]) != 0:
				last_time = np.sort(np.array(list(ui_collect_stat[uik][0])))[-1]
				ui_collect_feature[uik][2] = 3600/(label_day-last_time).total_seconds()
			if ui_collect_feature[uik][0] > 0 and ui_buy_feature[uik][0] == 0:
				ui_collect_feature[uik][3] = 1
			if ui_collect_feature[uik][0] == 0 and ui_buy_feature[uik][0] > 0:
				ui_collect_feature[uik][3] = 2
			if ui_collect_feature[uik][0] > 0 and ui_buy_feature[uik][0] > 0:
				ui_collect_feature[uik][3] = 3
			
			# cart
			ui_cart_feature[uik][1] = len(cart_day)
			if len(ui_cart_stat[uik][0]) != 0:
				last_cart_time = np.sort(np.array(list(ui_cart_stat[uik][0])))[-1]
				ui_cart_feature[uik][3] = 3600/(label_day-last_cart_time).total_seconds()
				if len(ui_buy_stat[uik][0]) != 0:
					last_buy_time = np.sort(np.array(list(ui_buy_stat[uik][0])))[-1]
					if last_cart_time.date() == sday.date() and last_buy_time >= last_cart_time:
						ui_cart_feature[uik][7] = 1
			ui_cart_feature[uik][4] = len(cart_day&buy_day)
			ui_cart_feature[uik][6] = len(cart_day_span)
			
			# buy
			ui_buy_feature[uik][2] = len(buy_day)
			if len(ui_buy_stat[uik][0]) != 0:
				last_time = np.sort(np.array(list(ui_buy_stat[uik][0])))[-1]
				ui_buy_feature[uik][0] = 3600/(label_day-last_time).total_seconds()
			ui_buy_feature[uik][5] = len(buy_day_span)
			# others


	del ui_click_stat
	del ui_collect_stat
	del ui_cart_stat
	del ui_buy_stat

	user_subset = set()
	item_sample_set = set()
	for uik in ui_set:
		user_subset.add(uik[0])
		item_sample_set.add(uik[1])

	user_feature = dict()
	user_stat = dict()
	user_count = dict()
	item_feature = dict()
	item_stat = dict()
	item_click_everyday = dict()
	item_buy_everyday = dict()
	total_days = (sday - start_day).days + 1
	n_user_feature = 13
	n_item_feature = 10
	print "extract user and item feature"
	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		time_diff = (date_time - start_day).days
		uik = (user,item)
		beh = int(e['behavior_type'])
		if user in user_subset and date_time.date() <= sday.date():
			if user not in user_feature:
				user_feature[user] = [0 for i in xrange(n_user_feature)]
				user_stat[user] = [set() for i in xrange(4)]
				user_count[user] = [0 for i in xrange(4)]
			if date_time.date() == sday.date():
				user_feature[user][beh-1] += 1
			user_count[user][beh-1] += 1
			user_stat[user][beh-1].add(date_time)
			

		if item in item_sample_set and date_time.date() <= sday.date():
			if item not in item_feature :
				item_feature[item] = [0 for i in xrange(n_item_feature)]
				item_stat[item] = [set() for i in xrange(4)]
				item_click_everyday[item] = np.zeros(total_days)
				item_buy_everyday[item] = np.zeros(total_days)
			if date_time.date() == sday.date():
				item_feature[item][beh-1] += 1
			item_stat[item][beh-1].add(date_time)
			item_click_everyday[item][time_diff] += 1
			item_buy_everyday[item][time_diff] += 1
			
		
	last3days_set = set([sday.date(),(sday-timedelta(1)).date(),(sday-timedelta(2)).date()])
	time_x = np.arange(total_days).reshape(total_days,1)

	for user in user_subset:
		user_click_day = set([d.date() for d in user_stat[user][0]])
		user_cart_day = set([d.date() for d in user_stat[user][2]])
		user_buy_day = set([d.date() for d in user_stat[user][3]])
		if len(user_stat[user][0]) != 0:
			last_user_click_time = np.sort(np.array(list(user_stat[user][0])))[-1]
			user_feature[user][4] = 3600/(label_day-last_user_click_time).total_seconds()
		if len(user_stat[user][1]) != 0:
			last_user_collect_time = np.sort(np.array(list(user_stat[user][1])))[-1]
			user_feature[user][5] = 3600/(label_day-last_user_collect_time).total_seconds()
		if len(user_stat[user][2]) != 0:
			last_user_cart_time = np.sort(np.array(list(user_stat[user][2])))[-1]
			user_feature[user][6] = 3600/(label_day-last_user_cart_time).total_seconds()
		if len(user_stat[user][3]) != 0:
			last_user_buy_time = np.sort(np.array(list(user_stat[user][3])))[-1]
			user_feature[user][7] = 3600/(label_day-last_user_buy_time).total_seconds()
		user_feature[user][8] = (user_count[user][3]*1.0) / user_count[user][0] if user_count[user][0] != 0 else 0
		user_feature[user][9] = (user_count[user][3]*1.0) / user_count[user][2] if user_count[user][2] != 0 else 0
		user_feature[user][10] = len(user_click_day&last3days_set)
		user_feature[user][11] = len(user_cart_day&last3days_set)
		user_feature[user][12] = len(user_buy_day&last3days_set)
		

	for item in item_sample_set:
		if len(item_stat[item][0]) != 0:
			last_item_click_time = np.sort(np.array(list(item_stat[item][0])))[-1]
			item_feature[item][4] = 3600/(label_day-last_item_click_time).total_seconds()
		if len(item_stat[item][1]) != 0:
			last_item_click_time = np.sort(np.array(list(item_stat[item][1])))[-1]
			item_feature[item][5] = 3600/(label_day-last_item_click_time).total_seconds()
		if len(item_stat[item][2]) != 0:
			last_item_click_time = np.sort(np.array(list(item_stat[item][2])))[-1]
			item_feature[item][6] = 3600/(label_day-last_item_click_time).total_seconds()
		if len(item_stat[item][3]) != 0:
			last_item_click_time = np.sort(np.array(list(item_stat[item][3])))[-1]
			item_feature[item][7] = 3600/(label_day-last_item_click_time).total_seconds()

		lin_r = LinearRegression()
		lin_r.fit(time_x,item_click_everyday[item])
		item_feature[item][8] = lin_r.coef_

		lin_r = LinearRegression()
		lin_r.fit(time_x,item_buy_everyday[item])
		item_feature[item][9] = lin_r.coef_


	print "Write to file"
	# f_label = open(out_label,'w')
	# f_label.write('user_id,item_id\n')
	# for uik in ui_label:
	# 	f_label.write(uik[0]+','+uik[1]+'\n')
	# f_label.close()

	f_feature = open(out_feature,'w')
	f_feature.write('user_id,item_id,')
	for i in xrange(n_user_feature):
		f_feature.write('uf'+str(i)+',')
	for i in xrange(n_item_feature):
		f_feature.write('if'+str(i)+',')
	for i in xrange(n_click_feature):
		f_feature.write('uicl'+str(i)+',')
	for i in xrange(n_collect_feature):
		f_feature.write('uico'+str(i)+',')
	for i in xrange(n_cart_feature):
		f_feature.write('uica'+str(i)+',')
	for i in xrange(n_buy_featuure):
		if i != n_buy_featuure-1:
			f_feature.write('uibu'+str(i)+',')
		else:
			f_feature.write('uibu'+str(i)+'\n')

	for uik in ui_set:
		f_feature.write(uik[0]+','+uik[1]+',')
		for feat in user_feature[uik[0]]:
			f_feature.write(str(feat)+',')
		for feat in item_feature[uik[1]]:
			f_feature.write(str(feat)+',')
		for feat in ui_click_feature[uik]:
			f_feature.write(str(feat)+',')
		for feat in ui_collect_feature[uik]:
			f_feature.write(str(feat)+',')
		for feat in ui_cart_feature[uik]:
			f_feature.write(str(feat)+',')
		for feat in ui_buy_feature[uik]:
			f_feature.write(str(feat)+',')
		if uik in ui_label:
			f_feature.write('1\n')
		else:
			f_feature.write('0\n')
	f_feature.close()
