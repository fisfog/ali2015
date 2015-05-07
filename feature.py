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



def ex_behav_feature(infile,label_file,sub_item_file,outfile,last_day):
	print "extract item subset"
	item_subset = set()
	item_feature = dict()
	item_cate_dict = dict()
	cc = 0
	for e in util.parse(sub_item_file):
		item_subset.add(e['item_id'])
		if e['item_category'] not in item_cate_dict:
			item_cate_dict[e['item_category']] = cc
			cc += 1
		if e['item_id'] not in item_feature:
			item_feature[e['item_id']] = [item_cate_dict[e['item_category']]]
	print "#Item subset:%d"%(len(item_subset))

	# user_item_set = set()
	# for e in util.parse(infile):
	# 	user_item_set.add((e['user_id'],e['item_id']))
	# print "#(user,item) pair:%d"%(len(user_item_set))

	# print "allocate user,item behavior feature"
	# user_item_behav_feature = dict()
	# user_item_behav_stat = dict()
	# for uik in user_item_set:
	# 	user_item_behav_feature[uik] = [0 for i in xrange(32)]
	# 	user_item_behav_stat[uik] = [set() for i in xrange(3)]

	user_item_behav_feature = dict()
	user_item_behav_stat = dict()
	print "extract user,item behavior feature"
	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		time_int = util.tran_day(date_time)
		beh = int(e['behavior_type'])-1
		ui_key = (user,item)
		start_day = last_day - 7
		if item in item_subset:
			if ui_key not in user_item_behav_feature:
				user_item_behav_feature[ui_key] = [0 for i in xrange(32)]
				user_item_behav_stat[ui_key] = [set() for i in xrange(3)]
			if beh == 0:
				user_item_behav_feature[ui_key][0] += 1
				user_item_behav_stat[ui_key][0].add(time_int)
				if date_time.month == 12 and date_time.day >= start_day:
					user_item_behav_feature[ui_key][1] += 1
				if date_time.month == 12 and date_time.day == last_day:
					user_item_behav_feature[ui_key][2] += 1

			if beh == 1:
				user_item_behav_feature[ui_key][9] += 1
				if date_time.month == 12 and date_time.day >= start_day:
					user_item_behav_feature[ui_key][10] = 1
				if date_time.month == 12 and date_time.day == last_day:
					user_item_behav_feature[ui_key][11] = 1

			if beh == 2:
				user_item_behav_feature[ui_key][17] += 1
				user_item_behav_stat[ui_key][1].add(time_int)
				if date_time.month == 12 and date_time.day >= start_day:
					user_item_behav_feature[ui_key][18] += 1
				if date_time.month == 12 and date_time.day == last_day:
					user_item_behav_feature[ui_key][19] += 1

			if beh == 3:
				user_item_behav_feature[ui_key][25] += 1
				user_item_behav_stat[ui_key][2].add(time_int)
				if date_time.month == 12 and date_time.day >= start_day:
					user_item_behav_feature[ui_key][26] += 1
				if date_time.month == 12 and date_time.day == last_day:
					user_item_behav_feature[ui_key][27] += 1


	print "proc user,item behavior feature"
	for uik in user_item_behav_feature:
		# click feature
		user_item_behav_feature[uik][3] = len(user_item_behav_stat[uik][0])
		if user_item_behav_feature[uik][3] != 0:
			user_item_behav_feature[uik][4] = 1.0*user_item_behav_feature[uik][0] / user_item_behav_feature[uik][3]
		# 最近点击时间差
		if len(user_item_behav_stat[uik][0]) != 0:
			max_click_day = np.sort(np.array(list(user_item_behav_stat[uik][0])))[-1]
			max_click_d = max_click_day%100
			max_click_m = max_click_day/100%100
			diff = last_day - max_click_d + 31 * (12 - max_click_m) + 1
			user_item_behav_feature[uik][5] = 1.0 / diff
		user_item_behav_feature[uik][6] = len(user_item_behav_stat[uik][0]&user_item_behav_stat[uik][2])
		if user_item_behav_feature[uik][25] != 0:
			user_item_behav_feature[uik][7] = 1.0*user_item_behav_feature[uik][0] / user_item_behav_feature[uik][25]
		if user_item_behav_feature[uik][27] != 0:
			user_item_behav_feature[uik][8] = 1.0*user_item_behav_feature[uik][2] / user_item_behav_feature[uik][27]

		# collect feature
		user_item_behav_feature[uik][12] = 1 if user_item_behav_feature[uik][9] > 1 else 0
		if user_item_behav_feature[uik][9] > 0 and user_item_behav_feature[uik][25] > 0:
			user_item_behav_feature[uik][13] = 1
		if user_item_behav_feature[uik][12] != 0 and user_item_behav_feature[uik][25] > 0:
			user_item_behav_feature[uik][14] = 1
		if user_item_behav_feature[uik][10] != 0 and user_item_behav_feature[uik][26] > 0:
			user_item_behav_feature[uik][15] = 1
		if user_item_behav_feature[uik][11] != 0 and user_item_behav_feature[uik][27] > 0:
			user_item_behav_feature[uik][16] = 1


		# cart feature
		user_item_behav_feature[uik][20] = len(user_item_behav_stat[uik][1])
		if user_item_behav_feature[uik][20] != 0:
			user_item_behav_feature[uik][21] = 1.0*user_item_behav_feature[uik][17] / user_item_behav_feature[uik][20]
		user_item_behav_feature[uik][22] = len(user_item_behav_stat[uik][1]&user_item_behav_stat[uik][2])
		if user_item_behav_feature[uik][18] > 0 and user_item_behav_feature[uik][26] > 0:
			user_item_behav_feature[uik][23] = 1
		if user_item_behav_feature[uik][19] > 0 and user_item_behav_feature[uik][27] > 0:
			user_item_behav_feature[uik][24] = 1	

		# buy feature
		user_item_behav_feature[uik][28] = len(user_item_behav_stat[uik][2])
		if user_item_behav_feature[uik][28] != 0:
			user_item_behav_feature[uik][29] = 1.0*user_item_behav_feature[uik][25] / user_item_behav_feature[uik][28]
		if len(user_item_behav_stat[uik][2]) != 0:
			max_buy_day = np.sort(np.array(list(user_item_behav_stat[uik][2])))[-1]
			max_buy_d = max_buy_day%100
			max_buy_m = max_buy_day/100%100
			diff = last_day - max_buy_d + 31 * (12 - max_buy_m) + 1
			user_item_behav_feature[uik][30] = 1.0 / diff
		avg_df = util.avg_time_diff(user_item_behav_stat[uik][2])
		user_item_behav_feature[uik][31] = 1.0/avg_df if avg_df != 0 else 0

	ui_label = set()
	for e in util.parse(label_file):
		ui_label.add((e['user_id'],e['item_id']))

	print "Write to file"
	f = open(outfile,'w')
	f.write('user_id,item_id,cl0,cl1,cl2,cl3,cl4,cl5,cl6,cl7,cl8,co0,co1,co2,co3,co4,co5,co6,co7,\
		ca0,ca1,ca2,ca3,ca4,ca5,ca6,ca7,bu0,bu1,bu2,bu3,bu4,bu5,bu6,if0,label\n')
	for uik in user_item_behav_feature:
		f.write(uik[0]+','+uik[1]+',')
		for feat in user_item_behav_feature[uik]:
			f.write(str(feat)+',')
		for feat in item_feature[uik[1]]:
			f.write(str(feat)+',')
		if uik in ui_label:
			f.write('1\n')
		else:
			f.write('0\n')
	f.close()


def ex_behav_user_item_feature(infile,label_file,sub_item_file,outfile,last_day):
	print "extract item subset"
	item_subset = set()
	for e in util.parse(sub_item_file):
		item_subset.add(e['item_id'])
	print "#Item subset:%d"%(len(item_subset))

	print "user and item geohash 39.1*19.5"
	user_4geo,item_4geo = util.ex_user_item_4geohash('tianchi_mobile_recommend_train_user.csv','tianchi_mobile_recommend_train_item.csv')
	print "user and item geohash 4.9*4.9"
	user_5geo,item_5geo = util.ex_user_item_5geohash('tianchi_mobile_recommend_train_user.csv','tianchi_mobile_recommend_train_item.csv')


	user_item_behav_feature = dict()
	user_item_behav_stat = dict()
	n_ui_beh_feature = 48
	print "extract user,item behavior feature"
	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		time_int = util.tran_day(date_time)
		beh = int(e['behavior_type'])-1
		ui_key = (user,item)
		start_day = last_day - 7
		if item in item_subset:
			if ui_key not in user_item_behav_feature:
				user_item_behav_feature[ui_key] = [0 for i in xrange(n_ui_beh_feature)]
				user_item_behav_stat[ui_key] = [set() for i in xrange(7)]
			if beh == 0:
				user_item_behav_feature[ui_key][0] += 1
				user_item_behav_stat[ui_key][0].add(time_int)
				if date_time.month == 12 and date_time.day >= start_day:
					user_item_behav_feature[ui_key][1] += 1
				if date_time.month == 12 and date_time.day == last_day:
					user_item_behav_feature[ui_key][2] += 1

			if beh == 1:
				user_item_behav_feature[ui_key][9] += 1
				if date_time.month == 12 and date_time.day >= start_day:
					user_item_behav_feature[ui_key][10] = 1
				if date_time.month == 12 and date_time.day == last_day:
					user_item_behav_feature[ui_key][11] = 1

			if beh == 2:
				user_item_behav_feature[ui_key][17] += 1
				user_item_behav_stat[ui_key][1].add(time_int)
				if date_time.month == 12 and date_time.day >= start_day:
					user_item_behav_feature[ui_key][18] += 1
				if date_time.month == 12 and date_time.day == last_day:
					user_item_behav_feature[ui_key][19] += 1

			if beh == 3:
				user_item_behav_feature[ui_key][25] += 1
				user_item_behav_stat[ui_key][2].add(time_int)
				# user_item_behav_stat[ui_key][3].add(date_time)
				if date_time.month == 12 and date_time.day >= start_day:
					user_item_behav_feature[ui_key][26] += 1
				if date_time.month == 12 and date_time.day == last_day:
					user_item_behav_feature[ui_key][27] += 1
				if date_time.month== 11:
					user_item_behav_feature[ui_key][45] += 1
				elif date_time.month== 12 and date_time.day == 1:
					user_item_behav_feature[ui_key][45] += 1
				elif date_time.day >= 2 and date_time.day < 9:
					user_item_behav_feature[ui_key][46] += 1
				elif date_time.day >=9:
					user_item_behav_feature[ui_key][47] += 1

	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		time_int = util.tran_day(date_time)
		beh = int(e['behavior_type'])-1
		ui_key = (user,item)
		if item in item_subset:
			if len(user_item_behav_stat[ui_key][2]) > 0:
				if beh == 0:
					tt = date_time
					tt['d'] += 1
					tti = util.tran_day(tt)
					if tti in user_item_behav_stat[ui_key][2]:
						user_item_behav_stat[ui_key][3].add(tti)
						if tt['h'] >= 18:
							user_item_behav_stat[ui_key][4].add(tti)

				if beh == 1:
					tt = date_time
					tt['d'] += 1
					tti = util.tran_day(tt)
					if tti in user_item_behav_stat[ui_key][2]:
						user_item_behav_feature[ui_key][39] = 1
						if tt['h'] >= 18:
							user_item_behav_feature[ui_key][40] =1

				if beh == 2:
					tt = date_time
					tt['d'] += 1
					tti = util.tran_day(tt)
					if tti in user_item_behav_stat[ui_key][2]:
						user_item_behav_stat[ui_key][5].add(tti)
						if tt['h'] >= 18:
							user_item_behav_stat[ui_key][6].add(tti)


	print "proc user,item behavior feature"
	for uik in user_item_behav_feature:
		# click feature
		user_item_behav_feature[uik][3] = len(user_item_behav_stat[uik][0])
		if user_item_behav_feature[uik][3] != 0:
			user_item_behav_feature[uik][4] = 1.0*user_item_behav_feature[uik][0] / user_item_behav_feature[uik][3]
		# 最近点击时间差
		if len(user_item_behav_stat[uik][0]) != 0:
			max_click_day = np.sort(np.array(list(user_item_behav_stat[uik][0])))[-1]
			max_click_d = max_click_day%100
			max_click_m = max_click_day/100%100
			diff = last_day - max_click_d + 31 * (12 - max_click_m) + 1
			user_item_behav_feature[uik][5] = 1.0 / diff
		user_item_behav_feature[uik][6] = len(user_item_behav_stat[uik][0]&user_item_behav_stat[uik][2])
		if user_item_behav_feature[uik][0] != 0:
			user_item_behav_feature[uik][7] = (1.0*user_item_behav_feature[uik][25]+1) / user_item_behav_feature[uik][0]
		if user_item_behav_feature[uik][27] != 0:
			user_item_behav_feature[uik][8] = 1.0*user_item_behav_feature[uik][2] / user_item_behav_feature[uik][27]
		avg_df = util.avg_time_diff(user_item_behav_stat[uik][0])
		user_item_behav_feature[uik][34] = 1.0/avg_df if avg_df != 0 else 0

		if len(user_item_behav_stat[uik][2]) != 0:
			user_item_behav_feature[uik][37] = 1.0*len(user_item_behav_stat[uik][3])/len(user_item_behav_stat[uik][2])
			user_item_behav_feature[uik][38] = 1.0*len(user_item_behav_stat[uik][4])/len(user_item_behav_stat[uik][2])

		# collect feature
		user_item_behav_feature[uik][12] = 1 if user_item_behav_feature[uik][9] > 1 else 0
		if user_item_behav_feature[uik][9] > 0 and user_item_behav_feature[uik][25] > 0:
			user_item_behav_feature[uik][13] = 1
		if user_item_behav_feature[uik][12] != 0 and user_item_behav_feature[uik][25] > 0:
			user_item_behav_feature[uik][14] = 1
		if user_item_behav_feature[uik][10] != 0 and user_item_behav_feature[uik][26] > 0:
			user_item_behav_feature[uik][15] = 1
		if user_item_behav_feature[uik][11] != 0 and user_item_behav_feature[uik][27] > 0:
			user_item_behav_feature[uik][16] = 1
		if user_item_behav_feature[uik][9] > 0 and user_item_behav_feature[uik][25] > 1:
			user_item_behav_feature[uik][32] = 1

		# cart feature
		user_item_behav_feature[uik][20] = len(user_item_behav_stat[uik][1])
		if user_item_behav_feature[uik][20] != 0:
			user_item_behav_feature[uik][21] = 1.0*user_item_behav_feature[uik][17] / user_item_behav_feature[uik][20]
		user_item_behav_feature[uik][22] = len(user_item_behav_stat[uik][1]&user_item_behav_stat[uik][2])
		if user_item_behav_feature[uik][18] > 0 and user_item_behav_feature[uik][26] > 0:
			user_item_behav_feature[uik][23] = 1
		if user_item_behav_feature[uik][19] > 0 and user_item_behav_feature[uik][27] > 0:
			user_item_behav_feature[uik][24] = 1
		if user_item_behav_feature[uik][17] > 0:
			user_item_behav_feature[uik][33] = (1.0*user_item_behav_feature[uik][25]+1) / user_item_behav_feature[uik][17]

		if len(user_item_behav_stat[uik][2]) != 0:
			user_item_behav_feature[uik][41] = 1.0*len(user_item_behav_stat[uik][5])/len(user_item_behav_stat[uik][2])
			user_item_behav_feature[uik][42] = 1.0*len(user_item_behav_stat[uik][6])/len(user_item_behav_stat[uik][2])

		# buy feature
		user_item_behav_feature[uik][28] = len(user_item_behav_stat[uik][2])
		if user_item_behav_feature[uik][28] != 0:
			user_item_behav_feature[uik][29] = 1.0*user_item_behav_feature[uik][25] / user_item_behav_feature[uik][28]
		if len(user_item_behav_stat[uik][2]) != 0:
			max_buy_day = np.sort(np.array(list(user_item_behav_stat[uik][2])))[-1]
			max_buy_d = max_buy_day%100
			max_buy_m = max_buy_day/100%100
			diff = last_day - max_buy_d + 31 * (12 - max_buy_m) + 1
			user_item_behav_feature[uik][30] = 1.0 / diff
		avg_df = util.avg_time_diff(user_item_behav_stat[uik][2])
		user_item_behav_feature[uik][31] = 1.0/avg_df if avg_df != 0 else 0

		min_df = util.min_time_diff(user_item_behav_stat[uik][2])
		user_item_behav_feature[uik][43] = 1.0/min_df if min_df != 0 else 0

		max_df = util.max_time_diff(user_item_behav_stat[uik][2])
		user_item_behav_feature[uik][44] = 1.0/max_df if max_df != 0 else 0

		user_item_behav_feature[uik][45] = user_item_behav_feature[uik][45]*1.0/15
		user_item_behav_feature[uik][46] = user_item_behav_feature[uik][46]*1.0/7
		user_item_behav_feature[uik][47] = user_item_behav_feature[uik][47]*1.0/(last_day-8)

		# other feature
		if len(user_4geo[uik[0]]) != 0 and len(item_4geo[uik[1]]) != 0:
			user_item_behav_feature[uik][35] = 1 if len(user_4geo[uik[0]]&item_4geo[uik[1]]) != 0 else 0
		else:
			user_item_behav_feature[uik][35] = -1
		if len(user_5geo[uik[0]]) != 0 and len(item_5geo[uik[1]]) != 0:
			user_item_behav_feature[uik][36] = 1 if len(user_5geo[uik[0]]&item_5geo[uik[1]]) != 0 else 0
		else:
			user_item_behav_feature[uik][36] = -1


	del user_item_behav_stat

	user_feature = dict()
	user_stat = dict()
	n_user_feature = 36
	print "extract user feature"
	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		time_int = util.tran_day(date_time)
		beh = int(e['behavior_type'])-1
		# ui_key = (user,item)
		# start_day = last_day - 7
		if user not in user_feature:
			user_feature[user] = [0 for i in xrange(n_user_feature)]
			user_stat[user] = [set() for i in xrange(22)]
		if beh == 0:
			user_stat[user][0].add(time_int)
			user_stat[user][1].add(item)
			if item in item_subset:
				user_stat[user][2].add(time_int)
				user_stat[user][3].add(item)
				if date_time.month== 11:
					user_stat[user][4].add(time_int)
				elif date_time.month== 12 and date_time.day == 1:
					user_stat[user][4].add(time_int)
				elif date_time.day >= 2 and date_time.day < 9:
					user_stat[user][5].add(time_int)
				elif date_time.day >=9:
					user_stat[user][6].add(time_int)

		if beh == 1:
			user_stat[user][7].add(item)
			if item in item_subset:
				user_stat[user][8].add(item)

		if beh == 2:
			user_stat[user][9].add(time_int)
			user_stat[user][10].add(item)
			if item in item_subset:
				user_stat[user][11].add(time_int)
				user_stat[user][12].add(item)

		if beh == 3:
			user_stat[user][13].add(time_int)
			user_stat[user][14].add(item)
			if item in item_subset:
				user_stat[user][15].add(time_int)
				if item in user_stat[user][16]:
					user_stat[user][20].add(item)
				user_stat[user][16].add(item)
				if date_time.month== 11:
					user_stat[user][17].add(time_int)
				elif date_time.month== 12 and date_time.day == 1:
					user_stat[user][17].add(time_int)
				elif date_time.day >= 2 and date_time.day < 9:
					user_stat[user][18].add(time_int)
				elif date_time.day >=9:
					user_stat[user][19].add(time_int)
				user_stat[user][21].add(e['item_category'])


	print "proc user feature"
	for user in user_feature:
		# click feature
		user_feature[user][0] = len(user_stat[user][0])
		user_feature[user][1] = len(user_stat[user][1])
		if user_feature[user][1] != 0:
			user_feature[user][2] = (1.0*user_feature[user][23]+1)/user_feature[user][1]
		if user_feature[user][0] != 0:
			user_feature[user][3] = 1.0*user_feature[user][1]/user_feature[user][0]
		user_feature[user][4] = len(user_stat[user][2])
		user_feature[user][5] = len(user_stat[user][3])
		if user_feature[user][5] != 0:
			user_feature[user][6] = (1.0*user_feature[user][26]+1)/user_feature[user][5]
		if user_feature[user][4] != 0:
			user_feature[user][7] = 1.0*user_feature[user][5]/user_feature[user][4]
		if user_feature[user][1] != 0:
			user_feature[user][8] = (1.0*user_feature[user][5]+1)/user_feature[user][1]

		# collect feature
		user_feature[user][9] = len(user_stat[user][7])
		user_feature[user][10] = len(user_stat[user][7]&user_stat[user][14])
		user_feature[user][11] = len(user_stat[user][8])
		user_feature[user][12] = len(user_stat[user][8]&user_stat[user][16])
		if user_feature[user][9] != 0:
			user_feature[user][13] = (1.0*user_feature[user][11]+1)/user_feature[user][9]

		# cart feature
		user_feature[user][14] = len(user_stat[user][9])
		user_feature[user][15] = len(user_stat[user][10])
		if user_feature[user][15] != 0:
			user_feature[user][16] = (1.0*user_feature[user][23]+1)/user_feature[user][15]
		if user_feature[user][14] != 0:
			user_feature[user][17] = 1.0*user_feature[user][15]/user_feature[user][14]

		user_feature[user][18] = len(user_stat[user][11])
		user_feature[user][19] = len(user_stat[user][12])
		if user_feature[user][19] != 0:
			user_feature[user][20] = (1.0*user_feature[user][26])/user_feature[user][19]
		if user_feature[user][15] != 0:
			user_feature[user][21] = (1.0*user_feature[user][19]+1)/user_feature[user][15]

		# buy feature
		user_feature[user][22] = len(user_stat[user][13])
		user_feature[user][23] = len(user_stat[user][14])
		if user_feature[user][22] != 0:
			user_feature[user][24] = 1.0*user_feature[user][23]/user_feature[user][22]
		user_feature[user][25] = len(user_stat[user][15])
		user_feature[user][26] = len(user_stat[user][16])
		if user_feature[user][26] != 0:
			user_feature[user][27] = (1.0*len(user_stat[user][16])+1)/user_feature[user][26]
		if user_feature[user][23] != 0:
			user_feature[user][28] = (1.0*user_feature[user][26]+1)/user_feature[user][23]
		user_feature[user][35] = len(user_stat[user][21])

		# other feature
		user_feature[user][29] = 1.0*len(user_stat[user][4])/15
		user_feature[user][30] = 1.0*len(user_stat[user][5])/7
		user_feature[user][31] = 1.0*len(user_stat[user][6])/(last_day-8)

		user_feature[user][32] = 1.0*len(user_stat[user][17])/14
		user_feature[user][33] = 1.0*len(user_stat[user][18])/7
		user_feature[user][34] = 1.0*len(user_stat[user][19])/(last_day-8)

	del user_stat

	item_feature = dict()
	item_stat = dict()
	n_item_feature = 13
	print "extract item feature"
	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		time_int = util.tran_day(date_time)
		beh = int(e['behavior_type'])-1
		ui_key = (user,item)
		start_day = last_day - 7
		if item in item_subset:
			if item not in item_feature:
				item_feature[item] = [0 for i in xrange(n_item_feature)]
				item_stat[item] = [set() for i in xrange(8)]
			if beh == 0:
				if user in item_stat[item][0]:
					item_stat[item][1].add(user)
				item_stat[item][0].add(user)

			if beh == 1:
				item_stat[item][2].add(user)

			if beh == 2:
				item_stat[item][3].add(user)
				if date_time.month == 12 and date_time.day == last_day:
					item_feature[item][11] += 1

			if beh == 3:
				if user in item_stat[item][4]:
					item_stat[item][5].add(user)
				item_stat[item][4].add(user)
				if date_time.month == 11:
					item_stat[item][6].add(user)
				if date_time.month == 12 and date_time.day == 1:
					item_stat[item][6].add(user)
				if date_time.month == 12 and date_time.day > 1:
					item_stat[item][7].add(user)
				if date_time.month == 12 and date_time.day == last_day:
					item_feature[item][12] += 1


	print "proc item feature"
	for item in item_feature:
		# click feature
		item_feature[item][0] = len(item_stat[item][0])
		if item_feature[item][0] != 0:
			item_feature[item][1] = 1.0*len(item_stat[item][1])/item_feature[item][0]
		if item_feature[item][0] != 0:
			item_feature[item][2] = (1.0*item_feature[item][7]+1)/item_feature[item][0]

		# collect feature
		item_feature[item][3] = len(item_stat[item][2])
		if item_feature[item][3] != 0:
			item_feature[item][4] = (1.0*item_feature[item][7]+1)/item_feature[item][3]

		# cart feature
		item_feature[item][5] = len(item_stat[item][3])
		if item_feature[item][5] != 0:
			item_feature[item][6] = (1.0*item_feature[item][7]+1)/item_feature[item][5]

		# buy feature
		item_feature[item][7] = len(item_stat[item][4])
		if item_feature[item][7] != 0:
			item_feature[item][8] = 1.0*len(item_stat[item][5])/item_feature[item][7]
		if len(item_stat[item][6]) != 0:
			item_feature[item][9] = 1.0*len(item_stat[item][7])/len(item_stat[item][6])

	del item_stat

	cc = 0
	item_cate_dict = dict()
	for e in util.parse(sub_item_file):
		if e['item_category'] not in item_cate_dict:
			item_cate_dict[e['item_category']] = cc
			cc += 1
		if e['item_id'] in item_feature:
			item_feature[e['item_id']][10] = item_cate_dict[e['item_category']]

	ui_label = set()
	for e in util.parse(label_file):
		ui_label.add((e['user_id'],e['item_id']))

	# ui_beh_set = set()

	print "Write to file"
	f = open(outfile,'w')
	# f.write('user_id,item_id,cl0,cl1,cl2,cl3,cl4,cl5,cl6,cl7,cl8,co0,co1,co2,co3,co4,co5,co6,co7,\
	# 	ca0,ca1,ca2,ca3,ca4,ca5,ca6,ca7,bu0,bu1,bu2,bu3,bu4,bu5,bu6,\
	# 	uf0,uf1,uf2,uf3,uf4,uf5,uf6,uf7,uf8,uf9,uf10,uf11,uf12,uf13,uf14,uf15,uf16,uf17,\
	# 	uf18,uf19,uf20,uf21,uf22,uf23,uf24,uf25,uf26,uf27,uf28,uf29,uf30,uf31,uf32,\
	# 	if0,if1,if2,if3,if4,if5,if6,if7,if8,if9,if10,if11,if12,if13,if14,if15,if16,if17,\
	# 	if18,if19,if20,if21,if22,if23,if24,if25,if26,if27,if28,if29,if30,if31,label\n')
	f.write('user_id,item_id,')
	for i in xrange(n_ui_beh_feature):
		f.write('beh_f'+str(i)+',')
	for i in xrange(n_user_feature):
		f.write('u_f'+str(i)+',')
	for i in xrange(n_item_feature):
		f.write('i_f'+str(i)+',')
	f.write('label\n')

	for uik in user_item_behav_feature:
		f.write(uik[0]+','+uik[1]+',')
		for feat in user_item_behav_feature[uik]:
			f.write(str(feat)+',')
		for feat in user_feature[uik[0]]:
			f.write(str(feat)+',')
		for feat in item_feature[uik[1]]:
			f.write(str(feat)+',')
		if uik in ui_label:
			f.write('1\n')
		else:
			f.write('0\n')
	f.close()


# extract ui behav feature
def ex_behav_user_item_feature_md(infile,label_file,sub_item_file,outfile,last_day):
	print "extract item subset"
	item_subset = set()
	for e in util.parse(sub_item_file):
		item_subset.add(e['item_id'])
	print "#Item subset:%d"%(len(item_subset))

	# print "user and item geohash 39.1*19.5"
	# user_4geo,item_4geo = util.ex_user_item_geohash('tianchi_mobile_recommend_train_user.csv','tianchi_mobile_recommend_train_item.csv',4)
	# print "user and item geohash 4.9*4.9"
	# user_5geo,item_5geo = util.ex_user_item_geohash('tianchi_mobile_recommend_train_user.csv','tianchi_mobile_recommend_train_item.csv',5)

	ui_label = set()
	for e in util.parse(label_file):
		ui_label.add((e['user_id'],e['item_id']))
	print "ui buy: %d"%(len(ui_label))

	print "extract total ui pairs"
	ui_set = set()
	for e in util.parse(infile):
		ui_set.add((e['user_id'],e['item_id']))
	print "total ui pair: %d"%(len(ui_set))

	print "sample ui_subset"
	ui_subset = set()
	for uik in ui_set:
		rd = np.random.random()
		if uik in ui_label or rd <= 0.1:
		# if uik[1] in item_subset:
			ui_subset.add(uik)
	# sample
	print "ui pair after sampled: %d"%(len(ui_subset))

	user_item_behav_feature = dict()
	user_item_behav_stat = dict()
	n_ui_beh_feature = 48
	print "extract user,item behavior feature"
	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		beh = int(e['behavior_type'])-1
		ui_key = (user,item)
		start_day = last_day - timedelta(7)
		if ui_key in ui_subset:
			if ui_key not in user_item_behav_feature:
				user_item_behav_feature[ui_key] = [0 for i in xrange(n_ui_beh_feature)]
				user_item_behav_stat[ui_key] = [set() for i in xrange(7)]
			if beh == 0:
				user_item_behav_feature[ui_key][0] += 1
				user_item_behav_stat[ui_key][0].add(date_time)
				if date_time.month == 12 and date_time.day >= start_day.day:
					user_item_behav_feature[ui_key][1] += 1
				if date_time.month == 12 and date_time.day == last_day.day:
					user_item_behav_feature[ui_key][2] += 1

			if beh == 1:
				user_item_behav_feature[ui_key][9] += 1
				if date_time.month == 12 and date_time.day >= start_day.day:
					user_item_behav_feature[ui_key][10] = 1
				if date_time.month == 12 and date_time.day == last_day.day:
					user_item_behav_feature[ui_key][11] = 1

			if beh == 2:
				user_item_behav_feature[ui_key][17] += 1
				user_item_behav_stat[ui_key][1].add(date_time)
				if date_time.month == 12 and date_time.day >= start_day.day:
					user_item_behav_feature[ui_key][18] += 1
				if date_time.month == 12 and date_time.day == last_day.day:
					user_item_behav_feature[ui_key][19] += 1

			if beh == 3:
				user_item_behav_feature[ui_key][25] += 1
				user_item_behav_stat[ui_key][2].add(date_time)
				# user_item_behav_stat[ui_key][3].add(date_time)
				if date_time.month == 12 and date_time.day >= start_day.day:
					user_item_behav_feature[ui_key][26] += 1
				if date_time.month == 12 and date_time.day == last_day.day:
					user_item_behav_feature[ui_key][27] += 1
				if date_time.month == 11:
					user_item_behav_feature[ui_key][45] += 1
				elif date_time.month == 12 and date_time.day == 1:
					user_item_behav_feature[ui_key][45] += 1
				elif date_time.day >= 2 and date_time.day < 9:
					user_item_behav_feature[ui_key][46] += 1
				elif date_time.day >=9:
					user_item_behav_feature[ui_key][47] += 1

	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		beh = int(e['behavior_type'])-1
		ui_key = (user,item)
		if ui_key in ui_subset:
			if len(user_item_behav_stat[ui_key][2]) > 0:
				if beh == 0:
					tt = (date_time + timedelta(1)).date()
					if tt in [k.date() for k in user_item_behav_stat[ui_key][2]]:
						user_item_behav_feature[ui_key][37] += 1
						if date_time.hour >= 20:
							user_item_behav_feature[ui_key][38] += 1
	
				if beh == 1:
					tt = (date_time + timedelta(1)).date()
					if tt in [k.date() for k in user_item_behav_stat[ui_key][2]]:
						user_item_behav_feature[ui_key][39] = 1
						if date_time.hour >= 20:
							user_item_behav_feature[ui_key][40] =1
	
				if beh == 2:
					tt = (date_time + timedelta(1)).date()
					if tt in [k.date() for k in user_item_behav_stat[ui_key][2]]:
						user_item_behav_stat[ui_key][3].add(date_time.date())
						if date_time.hour >= 20:
							user_item_behav_stat[ui_key][4].add(date_time.date())


	print "proc user,item behavior feature"
	for uik in user_item_behav_feature:
		# buy feature
		user_item_behav_feature[uik][28] = len(set([k.date() for k in user_item_behav_stat[uik][2]]))
		if user_item_behav_feature[uik][28] != 0:
			user_item_behav_feature[uik][29] = 1.0*user_item_behav_feature[uik][25] / user_item_behav_feature[uik][28]
		if len(user_item_behav_stat[uik][2]) != 0:
			max_buy_day = np.sort(np.array(list(user_item_behav_stat[uik][2])))[-1]
			diff = (last_day - max_buy_day).total_seconds()/3600
			user_item_behav_feature[uik][30] = 1.0 / (diff+1)
		avg_df = util.avg_time_diff(user_item_behav_stat[uik][2])
		user_item_behav_feature[uik][31] = 1.0/avg_df if avg_df != 0 else 0

		min_df = util.min_time_diff(user_item_behav_stat[uik][2])
		user_item_behav_feature[uik][43] = 1.0/min_df if min_df != 0 else 0

		max_df = util.max_time_diff(user_item_behav_stat[uik][2])
		user_item_behav_feature[uik][44] = 1.0/max_df if max_df != 0 else 0

		user_item_behav_feature[uik][45] = user_item_behav_feature[uik][45]*1.0/15
		user_item_behav_feature[uik][46] = user_item_behav_feature[uik][46]*1.0/7
		user_item_behav_feature[uik][47] = user_item_behav_feature[uik][47]*1.0/(last_day-datetime(2014,12,8)).days

		# click feature
		user_item_behav_feature[uik][3] = len(set([k.date() for k in user_item_behav_stat[uik][0]]))
		if user_item_behav_feature[uik][3] != 0:
			user_item_behav_feature[uik][4] = 1.0*user_item_behav_feature[uik][0] / user_item_behav_feature[uik][3]
		# 最近点击时间差
		if len(user_item_behav_stat[uik][0]) != 0:
			max_click_day = np.sort(np.array(list(user_item_behav_stat[uik][0])))[-1]
			diff = (last_day - max_click_day).total_seconds()/3600
			user_item_behav_feature[uik][5] = 1.0 / (diff+1) 
		user_item_behav_feature[uik][6] = len(user_item_behav_stat[uik][0]&user_item_behav_stat[uik][2])
		user_item_behav_feature[uik][7] = user_item_behav_feature[uik][0]/(1.0*user_item_behav_feature[uik][25]+1)
		if user_item_behav_feature[uik][27] != 0:
			user_item_behav_feature[uik][8] = 1.0*user_item_behav_feature[uik][2] / user_item_behav_feature[uik][27]
		avg_df = util.avg_time_diff(user_item_behav_stat[uik][0])
		user_item_behav_feature[uik][34] = 1.0/avg_df if avg_df != 0 else 0

		if user_item_behav_feature[uik][25] != 0:
			user_item_behav_feature[uik][37] = 1.0*user_item_behav_feature[uik][37]/user_item_behav_feature[uik][25]
			user_item_behav_feature[uik][38] = 1.0*user_item_behav_feature[uik][38]/user_item_behav_feature[uik][25]

		# collect feature
		user_item_behav_feature[uik][12] = 1 if user_item_behav_feature[uik][9] > 1 else 0
		if user_item_behav_feature[uik][9] > 0 and user_item_behav_feature[uik][25] > 0:
			user_item_behav_feature[uik][13] = 1
		if user_item_behav_feature[uik][12] != 0 and user_item_behav_feature[uik][25] > 0:
			user_item_behav_feature[uik][14] = 1
		if user_item_behav_feature[uik][10] != 0 and user_item_behav_feature[uik][26] > 0:
			user_item_behav_feature[uik][15] = 1
		if user_item_behav_feature[uik][11] != 0 and user_item_behav_feature[uik][27] > 0:
			user_item_behav_feature[uik][16] = 1
		if user_item_behav_feature[uik][9] > 0 and user_item_behav_feature[uik][25] > 1:
			user_item_behav_feature[uik][32] = 1

		# cart feature
		user_item_behav_feature[uik][20] = len(set([k.date() for k in user_item_behav_stat[uik][1]]))
		if user_item_behav_feature[uik][20] != 0:
			user_item_behav_feature[uik][21] = 1.0*user_item_behav_feature[uik][17] / user_item_behav_feature[uik][20]
		user_item_behav_feature[uik][22] = len(user_item_behav_stat[uik][1]&user_item_behav_stat[uik][2])
		if user_item_behav_feature[uik][18] > 0 and user_item_behav_feature[uik][26] > 0:
			user_item_behav_feature[uik][23] = 1
		if user_item_behav_feature[uik][19] > 0 and user_item_behav_feature[uik][27] > 0:
			user_item_behav_feature[uik][24] = 1
		if user_item_behav_feature[uik][17] > 0:
			user_item_behav_feature[uik][33] = (1.0*user_item_behav_feature[uik][25]+1) / user_item_behav_feature[uik][17]

		if len(set([k.date() for k in user_item_behav_stat[uik][2]])) != 0:
			user_item_behav_feature[uik][41] = 1.0*len(user_item_behav_stat[uik][3])/len(set([k.date() for k in user_item_behav_stat[uik][2]]))
			user_item_behav_feature[uik][42] = 1.0*len(user_item_behav_stat[uik][4])/len(set([k.date() for k in user_item_behav_stat[uik][2]]))


		# other feature
		# if len(user_4geo[uik[0]]) != 0 and len(item_4geo[uik[1]]) != 0:
		# 	user_item_behav_feature[uik][35] = 1 if len(user_4geo[uik[0]]&item_4geo[uik[1]]) != 0 else 0
		# else:
		# 	user_item_behav_feature[uik][35] = -1
		# if len(user_5geo[uik[0]]) != 0 and len(item_5geo[uik[1]]) != 0:
		# 	user_item_behav_feature[uik][36] = 1 if len(user_5geo[uik[0]]&item_5geo[uik[1]]) != 0 else 0
		# else:
		# 	user_item_behav_feature[uik][36] = -1

	del user_item_behav_stat

	user_subset = set()
	for uik in ui_subset:
		user_subset.add(uik[0])

	user_feature = dict()
	user_stat = dict()
	n_user_feature = 36
	print "extract user feature"
	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		beh = int(e['behavior_type'])-1
		# ui_key = (user,item)
		if user in user_subset:
			if user not in user_feature:
				user_feature[user] = [0 for i in xrange(n_user_feature)]
				user_stat[user] = [set() for i in xrange(22)]
			if beh == 0:
				user_stat[user][0].add(date_time.date())
				user_stat[user][1].add(item)
				if item in item_subset:
					user_stat[user][2].add(date_time.date())
					user_stat[user][3].add(item)
					if date_time.month== 11:
						user_stat[user][4].add(date_time.date())
					elif date_time.month== 12 and date_time.day == 1:
						user_stat[user][4].add(date_time.date())
					elif date_time.day >= 2 and date_time.day < 9:
						user_stat[user][5].add(date_time.date())
					elif date_time.day >=9:
						user_stat[user][6].add(date_time.date())

			if beh == 1:
				user_stat[user][7].add(item)
				if item in item_subset:
					user_stat[user][8].add(item)

			if beh == 2:
				user_stat[user][9].add(date_time.date())
				user_stat[user][10].add(item)
				if item in item_subset:
					user_stat[user][11].add(date_time.date())
					user_stat[user][12].add(item)

			if beh == 3:
				user_stat[user][13].add(date_time.date())
				user_stat[user][14].add(item)
				if item in item_subset:
					user_stat[user][15].add(date_time.date())
					if item in user_stat[user][16]:
						user_stat[user][20].add(item)
					user_stat[user][16].add(item)
					if date_time.month== 11:
						user_stat[user][17].add(date_time.date())
					elif date_time.month== 12 and date_time.day == 1:
						user_stat[user][17].add(date_time.date())
					elif date_time.day >= 2 and date_time.day < 9:
						user_stat[user][18].add(date_time.date())
					elif date_time.day >=9:
						user_stat[user][19].add(date_time.date())
					user_stat[user][21].add(e['item_category'])


	print "proc user feature"
	for user in user_feature:
		# buy feature
		user_feature[user][22] = len(user_stat[user][13])
		user_feature[user][23] = len(user_stat[user][14])
		if user_feature[user][22] != 0:
			user_feature[user][24] = 1.0*user_feature[user][23]/user_feature[user][22]
		user_feature[user][25] = len(user_stat[user][15])
		user_feature[user][26] = len(user_stat[user][16])
		if user_feature[user][26] != 0:
			user_feature[user][27] = (1.0*len(user_stat[user][16])+1)/user_feature[user][26]
		if user_feature[user][23] != 0:
			user_feature[user][28] = (1.0*user_feature[user][26]+1)/user_feature[user][23]
		user_feature[user][35] = len(user_stat[user][21])

		# click feature
		user_feature[user][0] = len(user_stat[user][0])
		user_feature[user][1] = len(user_stat[user][1])
		if user_feature[user][1] != 0:
			user_feature[user][2] = (1.0*user_feature[user][23]+1)/user_feature[user][1]
		if user_feature[user][0] != 0:
			user_feature[user][3] = 1.0*user_feature[user][1]/user_feature[user][0]
		user_feature[user][4] = len(user_stat[user][2])
		user_feature[user][5] = len(user_stat[user][3])
		if user_feature[user][5] != 0:
			user_feature[user][6] = (1.0*user_feature[user][26]+1)/user_feature[user][5]
		if user_feature[user][4] != 0:
			user_feature[user][7] = 1.0*user_feature[user][5]/user_feature[user][4]
		if user_feature[user][1] != 0:
			user_feature[user][8] = (1.0*user_feature[user][5]+1)/user_feature[user][1]

		# collect feature
		user_feature[user][9] = len(user_stat[user][7])
		user_feature[user][10] = len(user_stat[user][7]&user_stat[user][14])
		user_feature[user][11] = len(user_stat[user][8])
		user_feature[user][12] = len(user_stat[user][8]&user_stat[user][16])
		if user_feature[user][9] != 0:
			user_feature[user][13] = (1.0*user_feature[user][11]+1)/user_feature[user][9]

		# cart feature
		user_feature[user][14] = len(user_stat[user][9])
		user_feature[user][15] = len(user_stat[user][10])
		if user_feature[user][15] != 0:
			user_feature[user][16] = (1.0*user_feature[user][23]+1)/user_feature[user][15]
		if user_feature[user][14] != 0:
			user_feature[user][17] = 1.0*user_feature[user][15]/user_feature[user][14]

		user_feature[user][18] = len(user_stat[user][11])
		user_feature[user][19] = len(user_stat[user][12])
		if user_feature[user][19] != 0:
			user_feature[user][20] = (1.0*user_feature[user][26])/user_feature[user][19]
		if user_feature[user][15] != 0:
			user_feature[user][21] = (1.0*user_feature[user][19]+1)/user_feature[user][15]

		
		# other feature
		user_feature[user][29] = 1.0*len(user_stat[user][4])/15
		user_feature[user][30] = 1.0*len(user_stat[user][5])/7
		user_feature[user][31] = 1.0*len(user_stat[user][6])/(last_day-datetime(2014,12,8)).days

		user_feature[user][32] = 1.0*len(user_stat[user][17])/14
		user_feature[user][33] = 1.0*len(user_stat[user][18])/7
		user_feature[user][34] = 1.0*len(user_stat[user][19])/(last_day-datetime(2014,12,8)).days

	del user_stat

	item_sample_set = set()
	for uik in ui_subset:
		item_sample_set.add(uik[1])

	item_feature = dict()
	item_stat = dict()
	n_item_feature = 13
	print "extract item feature"
	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		beh = int(e['behavior_type'])-1
		ui_key = (user,item)
		start_day = last_day - timedelta(7)
		if item in item_sample_set:
			if item not in item_feature:
				item_feature[item] = [0 for i in xrange(n_item_feature)]
				item_stat[item] = [set() for i in xrange(8)]
			if beh == 0:
				if user in item_stat[item][0]:
					item_stat[item][1].add(user)
				item_stat[item][0].add(user)
	
			if beh == 1:
				item_stat[item][2].add(user)
	
			if beh == 2:
				item_stat[item][3].add(user)
				if date_time.month == 12 and date_time.day == last_day.day:
					item_feature[item][11] += 1
	
			if beh == 3:
				if user in item_stat[item][4]:
					item_stat[item][5].add(user)
				item_stat[item][4].add(user)
				if date_time.month == 11:
					item_stat[item][6].add(user)
				if date_time.month == 12 and date_time.day == 1:
					item_stat[item][6].add(user)
				if date_time.month == 12 and date_time.day > 1:
					item_stat[item][7].add(user)
				if date_time.month == 12 and date_time.day == last_day.day:
					item_feature[item][12] += 1
			item_feature[item][10] = int(e['item_category'])


	print "proc item feature"
	for item in item_feature:
		# click feature
		item_feature[item][0] = len(item_stat[item][0])
		if item_feature[item][0] != 0:
			item_feature[item][1] = 1.0*len(item_stat[item][1])/item_feature[item][0]
		if item_feature[item][0] != 0:
			item_feature[item][2] = (1.0*item_feature[item][7]+1)/item_feature[item][0]

		# collect feature
		item_feature[item][3] = len(item_stat[item][2])
		if item_feature[item][3] != 0:
			item_feature[item][4] = (1.0*item_feature[item][7]+1)/item_feature[item][3]

		# cart feature
		item_feature[item][5] = len(item_stat[item][3])
		if item_feature[item][5] != 0:
			item_feature[item][6] = (1.0*item_feature[item][7]+1)/item_feature[item][5]

		# buy feature
		item_feature[item][7] = len(item_stat[item][4])
		if item_feature[item][7] != 0:
			item_feature[item][8] = 1.0*len(item_stat[item][5])/item_feature[item][7]
		if len(item_stat[item][6]) != 0:
			item_feature[item][9] = 1.0*len(item_stat[item][7])/len(item_stat[item][6])

	del item_stat


	# ui_beh_set = set()

	print "Write to file"
	f = open(outfile,'w')
	# f.write('user_id,item_id,cl0,cl1,cl2,cl3,cl4,cl5,cl6,cl7,cl8,co0,co1,co2,co3,co4,co5,co6,co7,\
	# 	ca0,ca1,ca2,ca3,ca4,ca5,ca6,ca7,bu0,bu1,bu2,bu3,bu4,bu5,bu6,\
	# 	uf0,uf1,uf2,uf3,uf4,uf5,uf6,uf7,uf8,uf9,uf10,uf11,uf12,uf13,uf14,uf15,uf16,uf17,\
	# 	uf18,uf19,uf20,uf21,uf22,uf23,uf24,uf25,uf26,uf27,uf28,uf29,uf30,uf31,uf32,\
	# 	if0,if1,if2,if3,if4,if5,if6,if7,if8,if9,if10,if11,if12,if13,if14,if15,if16,if17,\
	# 	if18,if19,if20,if21,if22,if23,if24,if25,if26,if27,if28,if29,if30,if31,label\n')
	f.write('user_id,item_id,')
	for i in xrange(n_ui_beh_feature):
		f.write('beh_f'+str(i)+',')
	for i in xrange(n_user_feature):
		f.write('u_f'+str(i)+',')
	for i in xrange(n_item_feature):
		f.write('i_f'+str(i)+',')
	f.write('label\n')

	for uik in ui_subset:
		f.write(uik[0]+','+uik[1]+',')
		for feat in user_item_behav_feature[uik]:
			f.write(str(feat)+',')
		for feat in user_feature[uik[0]]:
			f.write(str(feat)+',')
		for feat in item_feature[uik[1]]:
			f.write(str(feat)+',')
		if uik in ui_label:
			f.write('1\n')
		else:
			f.write('0\n')
	f.close()




def ex_behav_user_item_feature(infile,item_subset,sday,outfile):
	# print "user and item geohash 39.1*19.5"
	# user_4geo,item_4geo = util.ex_user_item_geohash('tianchi_mobile_recommend_train_user.csv','tianchi_mobile_recommend_train_item.csv',4)
	# print "user and item geohash 4.9*4.9"
	# user_5geo,item_5geo = util.ex_user_item_geohash('tianchi_mobile_recommend_train_user.csv','tianchi_mobile_recommend_train_item.csv',5)

	print "extract total ui pairs"
	ui_set = set()
	for e in util.parse(infile):
		ui_set.add((e['user_id'],e['item_id']))
	print "total ui pair: %d"%(len(ui_set))

	print "sample ui_subset"
	ui_subset = set()
	for uik in ui_set:
		# rd = np.random.random()
		# if rd <= 0.1:
		if uik[1] in item_subset:
			ui_subset.add(uik)
	# sample
	print "ui pair after sampled: %d"%(len(ui_subset))

	label_day = sday + timedelta(1)
	last_week_start_day = sday - timedelta(7)
	ui_label = set()

	user_item_behav_feature = dict()
	user_item_behav_stat = dict()
	n_ui_beh_feature = 48
	print "extract user,item behavior feature"
	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		beh = int(e['behavior_type'])-1
		uik = (user,item)	
		if uik in ui_subset and date_time.date() <= sday.date():
			if uik not in user_item_behav_feature:
				user_item_behav_feature[uik] = [0 for i in xrange(n_ui_beh_feature)]
				user_item_behav_stat[uik] = [set() for i in xrange(7)]
			if beh == 0:
				user_item_behav_feature[uik][0] += 1
				user_item_behav_stat[uik][0].add(date_time)
				if date_time.date() >= last_week_start_day.date():
					user_item_behav_feature[uik][1] += 1
				if date_time.date() == sday.date():
					user_item_behav_feature[uik][2] += 1

			if beh == 1:
				user_item_behav_feature[uik][9] += 1
				if date_time.date() >= last_week_start_day.date():
					user_item_behav_feature[uik][10] = 1
				if date_time.date() == sday.date():
					user_item_behav_feature[uik][11] = 1

			if beh == 2:
				user_item_behav_feature[uik][17] += 1
				user_item_behav_stat[uik][1].add(date_time)
				if date_time.date() >= last_week_start_day.date():
					user_item_behav_feature[uik][18] += 1
				if date_time.date() == sday.date():
					user_item_behav_feature[uik][19] += 1

			if beh == 3:
				user_item_behav_feature[uik][25] += 1
				user_item_behav_stat[uik][2].add(date_time)
				# user_item_behav_stat[uik][3].add(date_time)
				if date_time.date() >= last_week_start_day.date():
					user_item_behav_feature[uik][26] += 1
				if date_time.date() == sday.date():
					user_item_behav_feature[uik][27] += 1
				if date_time.month == 11:
					user_item_behav_feature[uik][45] += 1
				elif date_time.month == 12 and date_time.day == 1:
					user_item_behav_feature[uik][45] += 1
				elif date_time.day >= 2 and date_time.day < 9:
					user_item_behav_feature[uik][46] += 1
				elif date_time.day >=9:
					user_item_behav_feature[uik][47] += 1

	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		beh = int(e['behavior_type'])-1
		uik = (user,item)
		if uik in ui_subset:
			if len(user_item_behav_stat[uik][2]) > 0:
				if beh == 0:
					tt = (date_time + timedelta(1)).date()
					if tt in [k.date() for k in user_item_behav_stat[uik][2]]:
						user_item_behav_feature[uik][37] += 1
						if date_time.hour >= 20:
							user_item_behav_feature[uik][38] += 1
	
				if beh == 1:
					tt = (date_time + timedelta(1)).date()
					if tt in [k.date() for k in user_item_behav_stat[uik][2]]:
						user_item_behav_feature[uik][39] = 1
						if date_time.hour >= 20:
							user_item_behav_feature[uik][40] =1
	
				if beh == 2:
					tt = (date_time + timedelta(1)).date()
					if tt in [k.date() for k in user_item_behav_stat[uik][2]]:
						user_item_behav_stat[uik][3].add(date_time.date())
						if date_time.hour >= 20:
							user_item_behav_stat[uik][4].add(date_time.date())


	print "proc user,item behavior feature"
	for uik in user_item_behav_feature:
		# buy feature
		user_item_behav_feature[uik][28] = len(set([k.date() for k in user_item_behav_stat[uik][2]]))
		if user_item_behav_feature[uik][28] != 0:
			user_item_behav_feature[uik][29] = 1.0*user_item_behav_feature[uik][25] / user_item_behav_feature[uik][28]
		if len(user_item_behav_stat[uik][2]) != 0:
			max_buy_day = np.sort(np.array(list(user_item_behav_stat[uik][2])))[-1]
			diff = (last_day - max_buy_day).total_seconds()/3600
			user_item_behav_feature[uik][30] = 1.0 / (diff+1)
		avg_df = util.avg_time_diff(user_item_behav_stat[uik][2])
		user_item_behav_feature[uik][31] = 1.0/avg_df if avg_df != 0 else 0

		min_df = util.min_time_diff(user_item_behav_stat[uik][2])
		user_item_behav_feature[uik][43] = 1.0/min_df if min_df != 0 else 0

		max_df = util.max_time_diff(user_item_behav_stat[uik][2])
		user_item_behav_feature[uik][44] = 1.0/max_df if max_df != 0 else 0

		user_item_behav_feature[uik][45] = user_item_behav_feature[uik][45]*1.0/15
		user_item_behav_feature[uik][46] = user_item_behav_feature[uik][46]*1.0/7
		user_item_behav_feature[uik][47] = user_item_behav_feature[uik][47]*1.0/(last_day-datetime(2014,12,8)).days

		# click feature
		user_item_behav_feature[uik][3] = len(set([k.date() for k in user_item_behav_stat[uik][0]]))
		if user_item_behav_feature[uik][3] != 0:
			user_item_behav_feature[uik][4] = 1.0*user_item_behav_feature[uik][0] / user_item_behav_feature[uik][3]
		# 最近点击时间差
		if len(user_item_behav_stat[uik][0]) != 0:
			max_click_day = np.sort(np.array(list(user_item_behav_stat[uik][0])))[-1]
			diff = (last_day - max_click_day).total_seconds()/3600
			user_item_behav_feature[uik][5] = 1.0 / (diff+1) 
		user_item_behav_feature[uik][6] = len(user_item_behav_stat[uik][0]&user_item_behav_stat[uik][2])
		user_item_behav_feature[uik][7] = user_item_behav_feature[uik][0]/(1.0*user_item_behav_feature[uik][25]+1)
		if user_item_behav_feature[uik][27] != 0:
			user_item_behav_feature[uik][8] = 1.0*user_item_behav_feature[uik][2] / user_item_behav_feature[uik][27]
		avg_df = util.avg_time_diff(user_item_behav_stat[uik][0])
		user_item_behav_feature[uik][34] = 1.0/avg_df if avg_df != 0 else 0

		if user_item_behav_feature[uik][25] != 0:
			user_item_behav_feature[uik][37] = 1.0*user_item_behav_feature[uik][37]/user_item_behav_feature[uik][25]
			user_item_behav_feature[uik][38] = 1.0*user_item_behav_feature[uik][38]/user_item_behav_feature[uik][25]

		# collect feature
		user_item_behav_feature[uik][12] = 1 if user_item_behav_feature[uik][9] > 1 else 0
		if user_item_behav_feature[uik][9] > 0 and user_item_behav_feature[uik][25] > 0:
			user_item_behav_feature[uik][13] = 1
		if user_item_behav_feature[uik][12] != 0 and user_item_behav_feature[uik][25] > 0:
			user_item_behav_feature[uik][14] = 1
		if user_item_behav_feature[uik][10] != 0 and user_item_behav_feature[uik][26] > 0:
			user_item_behav_feature[uik][15] = 1
		if user_item_behav_feature[uik][11] != 0 and user_item_behav_feature[uik][27] > 0:
			user_item_behav_feature[uik][16] = 1
		if user_item_behav_feature[uik][9] > 0 and user_item_behav_feature[uik][25] > 1:
			user_item_behav_feature[uik][32] = 1

		# cart feature
		user_item_behav_feature[uik][20] = len(set([k.date() for k in user_item_behav_stat[uik][1]]))
		if user_item_behav_feature[uik][20] != 0:
			user_item_behav_feature[uik][21] = 1.0*user_item_behav_feature[uik][17] / user_item_behav_feature[uik][20]
		user_item_behav_feature[uik][22] = len(user_item_behav_stat[uik][1]&user_item_behav_stat[uik][2])
		if user_item_behav_feature[uik][18] > 0 and user_item_behav_feature[uik][26] > 0:
			user_item_behav_feature[uik][23] = 1
		if user_item_behav_feature[uik][19] > 0 and user_item_behav_feature[uik][27] > 0:
			user_item_behav_feature[uik][24] = 1
		if user_item_behav_feature[uik][17] > 0:
			user_item_behav_feature[uik][33] = (1.0*user_item_behav_feature[uik][25]+1) / user_item_behav_feature[uik][17]

		if len(set([k.date() for k in user_item_behav_stat[uik][2]])) != 0:
			user_item_behav_feature[uik][41] = 1.0*len(user_item_behav_stat[uik][3])/len(set([k.date() for k in user_item_behav_stat[uik][2]]))
			user_item_behav_feature[uik][42] = 1.0*len(user_item_behav_stat[uik][4])/len(set([k.date() for k in user_item_behav_stat[uik][2]]))


		# other feature
		# if len(user_4geo[uik[0]]) != 0 and len(item_4geo[uik[1]]) != 0:
		# 	user_item_behav_feature[uik][35] = 1 if len(user_4geo[uik[0]]&item_4geo[uik[1]]) != 0 else 0
		# else:
		# 	user_item_behav_feature[uik][35] = -1
		# if len(user_5geo[uik[0]]) != 0 and len(item_5geo[uik[1]]) != 0:
		# 	user_item_behav_feature[uik][36] = 1 if len(user_5geo[uik[0]]&item_5geo[uik[1]]) != 0 else 0
		# else:
		# 	user_item_behav_feature[uik][36] = -1

	del user_item_behav_stat

	user_subset = set()
	for uik in ui_subset:
		user_subset.add(uik[0])

	user_feature = dict()
	user_stat = dict()
	n_user_feature = 36
	print "extract user feature"
	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		beh = int(e['behavior_type'])-1
		# ui_key = (user,item)
		if user in user_subset:
			if user not in user_feature:
				user_feature[user] = [0 for i in xrange(n_user_feature)]
				user_stat[user] = [set() for i in xrange(22)]
			if beh == 0:
				user_stat[user][0].add(date_time.date())
				user_stat[user][1].add(item)
				if item in item_subset:
					user_stat[user][2].add(date_time.date())
					user_stat[user][3].add(item)
					if date_time.month== 11:
						user_stat[user][4].add(date_time.date())
					elif date_time.month== 12 and date_time.day == 1:
						user_stat[user][4].add(date_time.date())
					elif date_time.day >= 2 and date_time.day < 9:
						user_stat[user][5].add(date_time.date())
					elif date_time.day >=9:
						user_stat[user][6].add(date_time.date())

			if beh == 1:
				user_stat[user][7].add(item)
				if item in item_subset:
					user_stat[user][8].add(item)

			if beh == 2:
				user_stat[user][9].add(date_time.date())
				user_stat[user][10].add(item)
				if item in item_subset:
					user_stat[user][11].add(date_time.date())
					user_stat[user][12].add(item)

			if beh == 3:
				user_stat[user][13].add(date_time.date())
				user_stat[user][14].add(item)
				if item in item_subset:
					user_stat[user][15].add(date_time.date())
					if item in user_stat[user][16]:
						user_stat[user][20].add(item)
					user_stat[user][16].add(item)
					if date_time.month== 11:
						user_stat[user][17].add(date_time.date())
					elif date_time.month== 12 and date_time.day == 1:
						user_stat[user][17].add(date_time.date())
					elif date_time.day >= 2 and date_time.day < 9:
						user_stat[user][18].add(date_time.date())
					elif date_time.day >=9:
						user_stat[user][19].add(date_time.date())
					user_stat[user][21].add(e['item_category'])


	print "proc user feature"
	for user in user_feature:
		# buy feature
		user_feature[user][22] = len(user_stat[user][13])
		user_feature[user][23] = len(user_stat[user][14])
		if user_feature[user][22] != 0:
			user_feature[user][24] = 1.0*user_feature[user][23]/user_feature[user][22]
		user_feature[user][25] = len(user_stat[user][15])
		user_feature[user][26] = len(user_stat[user][16])
		if user_feature[user][26] != 0:
			user_feature[user][27] = (1.0*len(user_stat[user][16])+1)/user_feature[user][26]
		if user_feature[user][23] != 0:
			user_feature[user][28] = (1.0*user_feature[user][26]+1)/user_feature[user][23]
		user_feature[user][35] = len(user_stat[user][21])

		# click feature
		user_feature[user][0] = len(user_stat[user][0])
		user_feature[user][1] = len(user_stat[user][1])
		if user_feature[user][1] != 0:
			user_feature[user][2] = (1.0*user_feature[user][23]+1)/user_feature[user][1]
		if user_feature[user][0] != 0:
			user_feature[user][3] = 1.0*user_feature[user][1]/user_feature[user][0]
		user_feature[user][4] = len(user_stat[user][2])
		user_feature[user][5] = len(user_stat[user][3])
		if user_feature[user][5] != 0:
			user_feature[user][6] = (1.0*user_feature[user][26]+1)/user_feature[user][5]
		if user_feature[user][4] != 0:
			user_feature[user][7] = 1.0*user_feature[user][5]/user_feature[user][4]
		if user_feature[user][1] != 0:
			user_feature[user][8] = (1.0*user_feature[user][5]+1)/user_feature[user][1]

		# collect feature
		user_feature[user][9] = len(user_stat[user][7])
		user_feature[user][10] = len(user_stat[user][7]&user_stat[user][14])
		user_feature[user][11] = len(user_stat[user][8])
		user_feature[user][12] = len(user_stat[user][8]&user_stat[user][16])
		if user_feature[user][9] != 0:
			user_feature[user][13] = (1.0*user_feature[user][11]+1)/user_feature[user][9]

		# cart feature
		user_feature[user][14] = len(user_stat[user][9])
		user_feature[user][15] = len(user_stat[user][10])
		if user_feature[user][15] != 0:
			user_feature[user][16] = (1.0*user_feature[user][23]+1)/user_feature[user][15]
		if user_feature[user][14] != 0:
			user_feature[user][17] = 1.0*user_feature[user][15]/user_feature[user][14]

		user_feature[user][18] = len(user_stat[user][11])
		user_feature[user][19] = len(user_stat[user][12])
		if user_feature[user][19] != 0:
			user_feature[user][20] = (1.0*user_feature[user][26])/user_feature[user][19]
		if user_feature[user][15] != 0:
			user_feature[user][21] = (1.0*user_feature[user][19]+1)/user_feature[user][15]

		
		# other feature
		user_feature[user][29] = 1.0*len(user_stat[user][4])/15
		user_feature[user][30] = 1.0*len(user_stat[user][5])/7
		user_feature[user][31] = 1.0*len(user_stat[user][6])/(last_day-datetime(2014,12,8)).days

		user_feature[user][32] = 1.0*len(user_stat[user][17])/14
		user_feature[user][33] = 1.0*len(user_stat[user][18])/7
		user_feature[user][34] = 1.0*len(user_stat[user][19])/(last_day-datetime(2014,12,8)).days

	del user_stat

	item_sample_set = set()
	for uik in ui_subset:
		item_sample_set.add(uik[1])

	item_feature = dict()
	item_stat = dict()
	n_item_feature = 13
	print "extract item feature"
	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		beh = int(e['behavior_type'])-1
		ui_key = (user,item)
		start_day = last_day - timedelta(7)
		if item in item_sample_set:
			if item not in item_feature:
				item_feature[item] = [0 for i in xrange(n_item_feature)]
				item_stat[item] = [set() for i in xrange(8)]
			if beh == 0:
				if user in item_stat[item][0]:
					item_stat[item][1].add(user)
				item_stat[item][0].add(user)
	
			if beh == 1:
				item_stat[item][2].add(user)
	
			if beh == 2:
				item_stat[item][3].add(user)
				if date_time.month == 12 and date_time.day == last_day.day:
					item_feature[item][11] += 1
	
			if beh == 3:
				if user in item_stat[item][4]:
					item_stat[item][5].add(user)
				item_stat[item][4].add(user)
				if date_time.month == 11:
					item_stat[item][6].add(user)
				if date_time.month == 12 and date_time.day == 1:
					item_stat[item][6].add(user)
				if date_time.month == 12 and date_time.day > 1:
					item_stat[item][7].add(user)
				if date_time.month == 12 and date_time.day == last_day.day:
					item_feature[item][12] += 1
			item_feature[item][10] = int(e['item_category'])


	print "proc item feature"
	for item in item_feature:
		# click feature
		item_feature[item][0] = len(item_stat[item][0])
		if item_feature[item][0] != 0:
			item_feature[item][1] = 1.0*len(item_stat[item][1])/item_feature[item][0]
		if item_feature[item][0] != 0:
			item_feature[item][2] = (1.0*item_feature[item][7]+1)/item_feature[item][0]

		# collect feature
		item_feature[item][3] = len(item_stat[item][2])
		if item_feature[item][3] != 0:
			item_feature[item][4] = (1.0*item_feature[item][7]+1)/item_feature[item][3]

		# cart feature
		item_feature[item][5] = len(item_stat[item][3])
		if item_feature[item][5] != 0:
			item_feature[item][6] = (1.0*item_feature[item][7]+1)/item_feature[item][5]

		# buy feature
		item_feature[item][7] = len(item_stat[item][4])
		if item_feature[item][7] != 0:
			item_feature[item][8] = 1.0*len(item_stat[item][5])/item_feature[item][7]
		if len(item_stat[item][6]) != 0:
			item_feature[item][9] = 1.0*len(item_stat[item][7])/len(item_stat[item][6])

	del item_stat


	# ui_beh_set = set()

	print "Write to file"
	f = open(outfile,'w')
	f.write('user_id,item_id,')
	for i in xrange(n_ui_beh_feature):
		f.write('beh_f'+str(i)+',')
	for i in xrange(n_user_feature):
		f.write('u_f'+str(i)+',')
	for i in xrange(n_item_feature):
		if i != n_item_feature-1:
			f.write('i_f'+str(i)+',')
		else:
			f.write('i_f'+str(i)+'\n')

	for uik in ui_subset:
		f.write(uik[0]+','+uik[1]+',')
		for feat in user_item_behav_feature[uik]:
			f.write(str(feat)+',')
		for feat in user_feature[uik[0]]:
			f.write(str(feat)+',')
		for i in xrange(n_item_feature):
			if i != n_item_feature-1:
				f.write(str(item_feature[uik[1]][i])+',')
			else:
				f.write(str(item_feature[uik[1]][i])+'\n')
	f.close()


def ex_oneday_train_feature(infile,item_subset,sday,out_feature,out_label):
	ui_beh_feature = dict()
	ui_beh_stat = dict()

	user_beh_item_category = dict()
	item_category = dict()
	ui_label = set()
	label_day = sday + timedelta(1)
	pre_day = sday - timedelta(1)
	n_ui_beh_feature = 9
	print "extract ui feature"
	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		uik = (user,item)
		beh = int(e['behavior_type'])
		if item not in item_subset:
			continue
		if user not in user_beh_item_category:
			user_beh_item_category[user] = set()
		if date_time.date() < sday.date():
			user_beh_item_category[user].add(e['item_category'])
		item_category[item] = e['item_category']
		if date_time.date() == label_day.date() and beh == 4:
			ui_label.add(uik)
		if date_time.date() == sday.date():
			if uik not in ui_beh_feature:
				ui_beh_feature[uik] = [0 for i in xrange(n_ui_beh_feature)]
				ui_beh_stat[uik] = [set() for i in xrange(4)]

			if beh == 1:
				ui_beh_feature[uik][0] += 1
				ui_beh_stat[uik][0].add(date_time)

			if beh == 2:
				ui_beh_feature[uik][1] = 1
				ui_beh_stat[uik][1].add(date_time)

			if beh == 3:
				ui_beh_feature[uik][2] += 1
				ui_beh_stat[uik][2].add(date_time)

			if beh == 4:
				ui_beh_feature[uik][3] += 1
				ui_beh_stat[uik][3].add(date_time)


	for uik in ui_beh_feature:
		# click
		if len(ui_beh_stat[uik][0]) != 0:
			last_time = np.sort(np.array(list(ui_beh_stat[uik][0])))[-1]
			ui_beh_feature[uik][4] = 3600/(label_day-last_time).total_seconds()

		# collect
		if len(ui_beh_stat[uik][1]) != 0:
			last_time = np.sort(np.array(list(ui_beh_stat[uik][1])))[-1]
			ui_beh_feature[uik][5] = 3600/(label_day-last_time).total_seconds()
		
		# cart
		if len(ui_beh_stat[uik][2]) != 0:
			last_time = np.sort(np.array(list(ui_beh_stat[uik][2])))[-1]
			ui_beh_feature[uik][6] = 3600/(label_day-last_time).total_seconds()
		
		# buy
		if len(ui_beh_stat[uik][3]) != 0:
			last_time = np.sort(np.array(list(ui_beh_stat[uik][3])))[-1]
			ui_beh_feature[uik][7] = 3600/(label_day-last_time).total_seconds()

		# others
		if item_category[uik[1]] in user_beh_item_category[uik[0]]:
			ui_beh_feature[uik][8] = 1

	print "Write to file"
	f_label = open(out_label,'w')
	f_label.write('user_id,item_id\n')
	for uik in ui_label:
		f_label.write(uik[0]+','+uik[1]+'\n')
	f_label.close()

	f_feature = open(out_feature,'w')
	f_feature.write('user_id,item_id,')
	for i in xrange(n_ui_beh_feature):
		if i != n_ui_beh_feature-1:
			f_feature.write('uif'+str(i)+',')
		else:
			f_feature.write('uif'+str(i)+'\n')
	for uik in ui_beh_feature:
		f_feature.write(uik[0]+','+uik[1]+',')
		for feat in ui_beh_feature[uik]:
			f_feature.write(str(feat)+',')
		if uik in ui_label:
			f_feature.write('1\n')
		else:
			f_feature.write('0\n')
	f_feature.close()

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
	time_span_a = sday - timedelta(2)
	time_span_b = sday - timedelta(2)
	start_day = datetime(2014,11,18)
	double12 = datetime(2014,12,20)

	n_click_feature = 10
	n_collect_feature = 5
	n_cart_feature = 9
	n_buy_featuure = 6
	print "extract ui feature"
	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		uik = (user,item)
		beh = int(e['behavior_type'])
		if item not in item_subset or date_time.date() == double12.date():
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
				if date_time.date() >= time_span_a.date() and date_time.date() <= sday.date():
					ui_click_feature[uik][6] += 1
	
			if beh == 2:
				ui_collect_feature[uik][0] += 1
				ui_collect_stat[uik][0].add(date_time)
				if date_time.date() == sday.date():
					ui_collect_feature[uik][1] += 1
				if date_time.date() >= time_span_a.date() and date_time.date() <= sday.date():
					ui_collect_feature[uik][4] = 1
	
			if beh == 3:
				ui_cart_feature[uik][0] += 1
				ui_cart_stat[uik][0].add(date_time)
				if date_time.date() == sday.date():
					ui_cart_feature[uik][2] += 1
				if date_time.date() >= time_span_a.date() and date_time.date() <= sday.date():
					ui_cart_feature[uik][5] += 1
	
			if beh == 4:
				ui_buy_feature[uik][0] += 1
				ui_buy_stat[uik][0].add(date_time)
				if date_time.date() == sday.date():
					ui_buy_feature[uik][2] += 1
				if date_time.date() >= time_span_a.date() and date_time.date() <= sday.date():
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
		ui_click_feature[uik][9] = (ui_buy_feature[uik][4] * 1.0) / ui_click_feature[uik][6] if ui_click_feature[uik][6] != 0 else 0

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
		ui_cart_feature[uik][8] = (ui_buy_feature[uik][4] * 1.0) / ui_cart_feature[uik][5] if ui_cart_feature[uik][5] != 0 else 0
		
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
	item_count = dict()
	# item_click_everyday = dict()
	# item_buy_everyday = dict()
	total_days = (sday - start_day).days + 1
	last3days_set = set([sday.date(),(sday-timedelta(1)).date(),(sday-timedelta(2)).date()])
	n_user_feature = 14
	n_item_feature = 14
	print "extract user and item feature"
	for e in util.parse(infile):
		user = e['user_id']
		item = e['item_id']
		date_time = util.time_proc(e['time'])
		# time_diff = (date_time - start_day).days
		uik = (user,item)
		beh = int(e['behavior_type'])
		if date_time.date == double12.date():
			continue
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
				item_stat[item] = [set() for i in xrange(8)]
				item_count[item] = [0 for i in xrange(4)]
				# item_click_everyday[item] = np.zeros(total_days)
				# item_buy_everyday[item] = np.zeros(total_days)
			if date_time.date() == sday.date():
				item_feature[item][beh-1] += 1
			item_stat[item][beh-1].add(date_time)
			if date_time.date() in last3days_set:
				item_stat[item][beh+3].add(user)
			item_count[item][beh-1] += 1
			# item_feature[item][14] = int(e['item_category'])
			# item_click_everyday[item][time_diff] += 1
			# item_buy_everyday[item][time_diff] += 1
			
		
	
	# time_x = np.arange(total_days).reshape(total_days,1)

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
		user_feature[user][13] = (user_count[user][3]*1.0) / user_count[user][1] if user_count[user][1] != 0 else 0
		user_feature[user][10] = len(user_click_day&last3days_set)
		user_feature[user][11] = len(user_cart_day&last3days_set)
		user_feature[user][12] = len(user_buy_day&last3days_set)
		# click_time_diff = util.avg_time_diff(user_stat[user][0])
		# user_feature[user][13] = 1.0 / click_time_diff if click_time_diff != 0 else 0
		# cart_time_diff = util.avg_time_diff(user_stat[user][2])
		# user_feature[user][14] = 1.0 / cart_time_diff if cart_time_diff != 0 else 0
		# buy_time_diff = util.avg_time_diff(user_stat[user][3])
		# user_feature[user][15] = 1.0 / buy_time_diff if buy_time_diff != 0 else 0

		

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
		item_feature[item][8] = len(item_stat[item][4])
		item_feature[item][9] = len(item_stat[item][5])
		item_feature[item][10] = len(item_stat[item][6])
		item_feature[item][11] = len(item_stat[item][7])
		item_feature[item][12] = (item_count[item][3]*1.0) / item_count[item][0] if item_count[item][0] != 0 else 0
		item_feature[item][13] = (item_count[item][3]*1.0) / item_count[item][2] if item_count[item][2] != 0 else 0


		# lin_r = LinearRegression()
		# lin_r.fit(time_x,item_click_everyday[item])
		# item_feature[item][8] = lin_r.coef_

		# lin_r = LinearRegression()
		# lin_r.fit(time_x,item_buy_everyday[item])
		# item_feature[item][9] = lin_r.coef_


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


# def append_meta_feature(infile,meta_feautre_file,sday,outfile)