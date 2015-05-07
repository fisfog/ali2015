#-*- coding:utf-8 -*-

"""
	littlekid
	muyunlei@gmail.com
"""

import util
import statdata
import feature


feature.ex_behav_user_item_feature_md('offline_train.csv','offline_train_label.csv','tianchi_mobile_recommend_train_item.csv','offline_train_sample_feature0.csv',datetime(2014,12,16,0,0))
feature.ex_behav_user_item_feature_md('offline_validation.csv','offline_val_label.csv','tianchi_mobile_recommend_train_item.csv','offline_val_sample_feature0.csv',datetime(2014,12,17,0,0))
feature.ex_behav_user_item_feature_md('offline_test.csv','offline_test_label.csv','tianchi_mobile_recommend_train_item.csv','offline_test_sample_feature0.csv',datetime(2014,12,18,0,0))