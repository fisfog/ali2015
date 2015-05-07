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
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier


class predict_model():
	# feature_should_scale = [2,3,4,5,8,11,19,20,21,22,24,27,28,29,30]
	def __init__(self,train_data,val_label,test_data):
		self.val_set = set()		
		for e in util.parse(val_label):
			self.val_set.add((int(e['user_id']),int(e['item_id'])))	
		self.train_data = sp.genfromtxt(train_data,delimiter=',',skip_header=1)
		self.val_data = sp.genfromtxt(test_data,delimiter=',',skip_header=1)
		feature_should_scale = [2,3,4,5,8,11,19,20,21,22,24,27,28,29,30]
		for i in feature_should_scale:
			self.train_data[:,i] = preprocessing.scale(self.train_data[:,i])
			self.val_data[:,i] = preprocessing.scale(self.val_data[:,i])		

		self.train_x = self.train_data[:,2:-2]
		self.train_y = self.train_data[:,-1]
		self.val_x = self.val_data[:,2:-2]
		self.val_y = self.val_data[:,-1]

		self.n_features = self.train_x.shape[1]

		self.neg_train_data = self.train_data[self.train_y==0]
		self.pos_train_data = self.train_data[self.train_y==1]
		self.user_geo,self.item_geo = util.ex_user_item_geohash('tianchi_mobile_recommend_train_user.csv','tianchi_mobile_recommend_train_item.csv')
		print "Total <user,item> pair: train = %d test = %d"%(len(self.train_data),len(self.val_data))

	def lr_fit(self,c,proportion,n_models=300):
		# self.n_models = self.neg_train_data.shape[0] / (slef.pos_train_data.shape[0]*10)
		self.n_models = n_models
		self.model_list = [LogisticRegression(C=c) for i in xrange(self.n_models)]

		for i in xrange(self.n_models):
			neg_sample_index = [np.random.randint(self.neg_train_data.shape[0]) for k in xrange(self.pos_train_data.shape[0]*proportion)]
			neg_sample_data = self.neg_train_data[neg_sample_index]
			new_data = np.array(neg_sample_data.tolist()+self.pos_train_data.tolist())
			np.random.shuffle(new_data)
			new_x = new_data[:,2:-2]
			new_y = new_data[:,-1]
			self.model_list[i].fit(new_x,new_y)
			print "model#%d Train Score: %f"%(i+1,self.model_list[i].score(new_x,new_y))

	def lr_predict(self,prob_threshold,vote_prop):
		result_vote = np.zeros(self.val_data.shape[0])
		for i in xrange(self.n_models):
			print "model#%d Validation Score: %f"%(i+1,self.model_list[i].score(self.val_x,self.val_y))
			result_vote += np.array(self.model_list[i].predict_proba(self.val_x)[:,1]>=prob_threshold,dtype=int)		
		pred = self.val_data[result_vote >= (self.n_models*vote_prop)]		
		result = set()
		for i in xrange(len(pred)):
			result.add((int(pred[i][0]),int(pred[i][1])))
		util.offline_f1(result,self.val_set)
		user_item_non_geo = set()
		for uik in result:
			user = str(uik[0])
			item = str(uik[1])
			if self.user_geo[user] != set() and self.item_geo[item] !=set():
				if self.user_geo[user]&self.item_geo[item] == set():
					user_item_non_geo.add(uik)
		for ui in user_item_non_geo:
			result.remove(ui)
		f1 = util.offline_f1(result,self.val_set)
		return f1

	def lr_aggregation(self,train_label,prob_threshold):
		train_result_set = set()
		for e in util.parse(train_label):
			train_result_set.add((int(e['user_id']),int(e['item_id'])))	

		f1_ay = np.zeros(self.n_models)
		for i in xrange(self.n_models):
			pred = self.train_data[self.model_list[i].predict_proba(self.train_x)[:,1]>=prob_threshold]
			result = set()
			for k in xrange(len(pred)):
				result.add((int(pred[k][0]),int(pred[k][1])))
			user_item_non_geo = set()
			for uik in result:
				user = str(uik[0])
				item = str(uik[1])
				if self.user_geo[user] != set() and self.item_geo[item] !=set():
					if self.user_geo[user]&self.item_geo[item] == set():
						user_item_non_geo.add(uik)
			for ui in user_item_non_geo:
				result.remove(ui)
			f1_ay[i] = util.offline_f1(result,train_result_set)
		# f1_ay = preprocessing.scale(f1_ay)
		self.lr_agg_coef = np.zeros(self.n_features)
		self.lr_agg_intercept = 0
		for i in xrange(self.n_models):
			coef = self.model_list[i].coef_.reshape(self.n_features)
			intercept = self.model_list[i].intercept_.reshape(1)
			self.lr_agg_coef += f1_ay[i] * coef
			self.lr_agg_intercept += f1_ay[i] * intercept
		self.lr_agg_coef /= np.sum(f1_ay)
		self.lr_agg_intercept /= np.sum(f1_ay)
		def sigmoid_decision(x):
			return 1.0 / (1+np.exp(-(np.dot(self.lr_agg_coef,x.T)+self.lr_agg_intercept)))
		lr_agg_pred = self.val_data[sigmoid_decision(self.val_x)>=prob_threshold]
		print 
		result = set()
		for k in xrange(len(pred)):
			result.add((int(pred[k][0]),int(pred[k][1])))
		util.offline_f1(result,self.val_set)
		user_item_non_geo = set()
		for uik in result:
			user = str(uik[0])
			item = str(uik[1])
			if self.user_geo[user] != set() and self.item_geo[item] !=set():
				if self.user_geo[user]&self.item_geo[item] == set():
					user_item_non_geo.add(uik)
		for ui in user_item_non_geo:
			result.remove(ui)
		f1 = util.offline_f1(result,self.val_set)
		return f1

