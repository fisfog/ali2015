#-*- coding:utf-8 -*-

"""
	littlekid
	muyunlei@gmail.com
"""

import util
import numpy as np
import scipy as sp
import time

class stat():
	"""
	统计类
	"""
	def __init__(self, filename):
		self.filename = filename
		self.user_id = {}
		self.item_id = {}
		self.id_user = {}
		self.id_item = {}
		# self.user_feature = []
		self.ucount = 0
		self.icount = 0
		self.recordN = 0
		for e in util.parse(self.filename):
			self.recordN += 1
			if e['user_id'] not in self.user_id:
				self.user_id[e['user_id']] = self.ucount
				self.id_user[self.ucount] = e['user_id']
				self.ucount += 1
			if e['item_id'] not in self.item_id:
				self.item_id[e['item_id']] = self.icount
				self.id_item[self.icount] = e['item_id']
				self.icount += 1
		print "#User:%d #Item:%d"%(len(self.user_id),len(self.item_id))

	def user_item_stat(self):
		"""
		统计user和item的各类特征
		维：0：浏览 1：收藏 2：购物车 3：购买
		"""
		self.user_feature = np.zeros((self.ucount,4))
		self.item_feature = np.zeros((self.icount,4))
		self.user_item_st = {}
		for e in util.parse(self.filename):
			u = e['user_id']
			i = e['item_id']
			b_t = int(e['behavior_type']) - 1
			self.user_feature[self.user_id[u]][b_t] += 1
			self.item_feature[self.item_id[i]][b_t] += 1
			if (u,i) not in self.user_item_st:
				self.user_item_st[(u,i)] = [0 for k in xrange(4)]
			self.user_item_st[(u,i)][b_t] += 1

	def plot_feature(self,bins=20):
		a = np.arange(0,max(ali_stat.user_feature[:,0])+1,max(ali_stat.user_feature[:,0])/bins)



