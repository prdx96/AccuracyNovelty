
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import time
import random
import tensorflow as tf
from scipy import spatial
import os
import json
import argparse
import scipy
import copy


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument('--beta', type=str, default='',help='input data path')
parser.add_argument('--iter', type=str, default='',help='input data path')
FLAGS, _ = parser.parse_known_args()

DATA_DIR='../../data/ml-100k/'
MODEL_DIR='./'

SEED=int(open("SEED.txt","r").readlines()[0])
DISTANT_TYPE=0
NOVELTY_TYPE=0
MLOBJ_PATH = 'ml_obj_%d.pkl'%(SEED)
UTILOBJ_PATH='ml_util_%d_dis_%d.pkl'%(SEED,DISTANT_TYPE)


# In[3]:


class DataSetProcesser():
    def calculate_data(self):
        self.list_uid = self.df_userinfo.uid
        self.list_itemid = self.df_iteminfo.itemid

        self.all_posuser_byitemid = {itemid: [] for itemid in self.list_itemid}
        self.all_positem_byuid = {uid: [] for uid in self.list_uid}
        self.all_neguser_byitemid = {itemid: [] for itemid in self.list_itemid}
        self.all_negitem_byuid = {uid: [] for uid in self.list_uid}
        sz1 = len(self.list_uid)
        sz2 = len(self.list_itemid)
        df_all = self.df_rating

        sz = len(self.df_rating)

        self.ratings_byitemid = [[0.0 for uid in self.list_uid]
                                 for itemid in self.list_itemid]

        for (index, row) in self.df_rating.iterrows():
            if index % 1000 == 0:
                print('Preprocessing Dataset', index, '/', sz)
            rating = row["rating"]
            uid = int(row["uid"])
            itemid = int(row["itemid"])

            self.ratings_byitemid[itemid][uid] = rating
            #self.rating_bypair[uid][itemid] = rating
            if rating > self.rating_threshold:
                self.all_posuser_byitemid[itemid].append(uid)
                self.all_positem_byuid[uid].append(itemid)
            else:
                self.all_neguser_byitemid[itemid].append(uid)
                self.all_negitem_byuid[uid].append(itemid)

        self._USER_SIZE_ONLY_NUM = len(self.user_numerical_attr)
        self._USER_SIZE_OF_FIELDS = []
        for feat in self.df_userinfo.columns:
            if feat not in self.user_numerical_attr:
                self._USER_SIZE_OF_FIELDS.append(
                    len(np.unique(self.df_userinfo[feat])))
        for feat in self.user_numerical_attr:
            self._USER_SIZE_OF_FIELDS.append(1)

        self._USER_SIZE = len(self._USER_SIZE_OF_FIELDS)
        self._USER_SIZE_OF_MASK_FIELDS = self._USER_SIZE_OF_FIELDS[:-self.
                                                                   _USER_SIZE_ONLY_NUM]
        self._USER_SIZE_BIN = sum(self._USER_SIZE_OF_FIELDS)

        self._ITEM_SIZE_ONLY_NUM = len(self.item_numerical_attr)

        self._ITEM_SIZE_OF_FIELDS = []
        for feat in self.df_iteminfo.columns:
            if feat in self.item_numerical_attr:
                self._ITEM_SIZE_OF_FIELDS.append(1)
            else:
                self._ITEM_SIZE_OF_FIELDS.append(
                    len(np.unique(self.df_iteminfo[feat])))

        self._ITEM_SIZE = len(self._ITEM_SIZE_OF_FIELDS)
        self._ITEM_SIZE_OF_MASK_FIELDS = self._ITEM_SIZE_OF_FIELDS[:-self.
                                                                   _ITEM_SIZE_ONLY_NUM]
        self._ITEM_SIZE_BIN = sum(self._ITEM_SIZE_OF_FIELDS)

    def split_dict(self, dic,ratio):
        seed = self.seed
        dic1 = {}
        dic2 = {}
        for ky in dic:
            lst = dic[ky]
            lenoflist = len(lst)
            if lenoflist != 0:
                random.Random(seed).shuffle(lst)
                dic1[ky] = lst[:int(ratio * lenoflist)]
                dic2[ky] = lst[int(ratio * lenoflist):]
            else:
                dic1[ky] = []
                dic2[ky] = []
        return dic1, dic2

    def merge_dict(self, dic1, dic2):
        return {ky: list(set(dic1[ky]) | set(dic2[ky])) for ky in dic1}

    def reverse_dict(self, dict_byuid):
        result = {itemid: [] for itemid in self.list_itemid}
        for uid in dict_byuid:
            for itemid in dict_byuid[uid]:
                result[itemid].append(uid)
        return result

    def split_data(self):
        self.train_positem_byuid, self.test_positem_byuid = self.split_dict(
            self.all_positem_byuid,self.ratio)

        self.train_posuser_byitemid, self.test_posuser_byitemid = self.reverse_dict(
            self.train_positem_byuid), self.reverse_dict(
                self.test_positem_byuid)

        self.train_negitem_byuid, self.test_negitem_byuid = self.split_dict(
            self.all_negitem_byuid,self.ratio)

        self.train_neguser_byitemid, self.test_neguser_byitemid = self.reverse_dict(
            self.train_negitem_byuid), self.reverse_dict(
                self.test_negitem_byuid)

        self.train_rateduser_byitemid = self.merge_dict(
            self.train_posuser_byitemid, self.train_neguser_byitemid)

        self.test_rateduser_byitemid = self.merge_dict(
            self.test_posuser_byitemid, self.test_neguser_byitemid)

        self.train_rateditem_byuid = self.merge_dict(self.train_positem_byuid,
                                                     self.train_negitem_byuid)

        self.test_rateditem_byuid = self.merge_dict(self.test_positem_byuid,
                                                    self.test_negitem_byuid)

    def __init__(self, movielens, split_ratio, seed=SEED):
        self.seed = seed
        self.rating_threshold = movielens.rating_threshold
        self.ratio = split_ratio
        self.df_rating = movielens.df_rating
        self.df_userinfo = movielens.df_userinfo
        self.df_iteminfo = movielens.df_iteminfo
        self.user_numerical_attr = movielens.user_numerical_attr
        self.item_numerical_attr = movielens.item_numerical_attr
        self.calculate_data()
        for attr in self.df_userinfo:
            if attr not in self.user_numerical_attr:
                #print(attr)
                #self.df_userinfo[attr] = self.df_userinfo[attr].astype('str')
                pass
            else:
                df = self.df_userinfo[attr].copy()
                self.df_userinfo.drop([attr], axis=1, inplace=True)
                self.df_userinfo[attr] = df
        for attr in self.df_iteminfo:
            if attr not in self.item_numerical_attr:
                #self.df_iteminfo[attr] = self.df_iteminfo[attr].astype('str')
                pass
            else:
                df = self.df_iteminfo[attr].copy()
                self.df_iteminfo.drop([attr], axis=1, inplace=True)
                self.df_iteminfo[attr] = df

        self.split_data()


# In[4]:


class MovieLens:
    def load_raw_data(self):
        f=tf.gfile.Open(DATA_DIR + 'u.data',"r")
        self.df_rating = pd.read_csv(
            f,
            sep='\t',
            names=['uid', 'itemid', 'rating', 'time'])
        
        f=tf.gfile.Open(DATA_DIR + 'u.user',"r")
        self.df_userinfo = pd.read_csv(
            f,
            sep='|',
            names=['uid', 'age', 'sex', 'occupation', 'zip_code'])
        list_item_attr = [
            'itemid', 'title', 'rel_date', 'video_rel_date', 'imdb_url',
            "unknown", "Action", "Adventure", "Animation", "Children's",
            "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
            "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
            "War", "Western"
        ]
        f=tf.gfile.Open(DATA_DIR + 'u.item',"r")
        self.df_iteminfo = pd.read_csv(
            f,
            sep='|',
            names=list_item_attr)
        self.df_userinfo = self.df_userinfo.fillna(0)
        self.df_iteminfo = self.df_iteminfo.fillna(0)

    def minmax_scaler(self, list_attr, df):
        for attr in list_attr:
            df[attr] = df[attr] - min(df[attr])
    def feature_engineering(self):

        ##iteminfo
        df_all = self.df_iteminfo
        df_date = df_all["rel_date"]
        df_date = pd.to_datetime(df_date)
        df_all["year"] = df_date.apply(lambda x: x.year)
        df_all["month"] = df_date.apply(lambda x: x.month)
        df_all["day"] = df_date.apply(lambda x: x.day)
        df_all.drop(
            ["rel_date", "imdb_url", "video_rel_date", "title"],
            axis=1,
            inplace=True)
        self.minmax_scaler(["year", "month", "day"], df_all)
        df_numeric = df_all.select_dtypes(exclude=['object'])
        df_obj = df_all.select_dtypes(include=['object']).copy()
        for c in df_obj:
            df_obj[c] = (pd.factorize(df_obj[c])[0])
        self.df_iteminfo = pd.concat([df_numeric, df_obj], axis=1)

        df_all = self.df_userinfo
        self.minmax_scaler(["age"], df_all)
        df_numeric = df_all.select_dtypes(exclude=['object'])
        df_obj = df_all.select_dtypes(include=['object']).copy()
        for c in df_obj:
            df_obj[c] = (pd.factorize(df_obj[c])[0])
        self.df_userinfo = pd.concat([df_numeric, df_obj], axis=1)

    def __init__(self):
        self.rating_threshold = 3
        self.load_raw_data()
        self.df_iteminfo["itemid"]=self.df_iteminfo["itemid"]-1
        self.df_userinfo["uid"]=self.df_userinfo["uid"]-1
        self.df_rating["itemid"]=self.df_rating["itemid"]-1
        self.df_rating["uid"]=self.df_rating["uid"]-1
        self.feature_engineering()
        
        self.user_numerical_attr =  ["age"]
        self.item_numerical_attr = ["year", "month", "day"]


# In[5]:


try:
    f=open(MLOBJ_PATH,"rb")
    dataset=pickle.load(f)
except:
    movielens=MovieLens()
    dataset=DataSetProcesser(movielens,0.7)
    f=open(MLOBJ_PATH,"wb")
    pickle.dump(dataset,f)


# In[6]:


def basic_stat():
    sz1=len(dataset.list_uid)
    sz2=len(dataset.list_itemid)
    sz3=len(dataset.df_rating)
    print(sz1,sz2,sz3,sz3/sz2/sz1)

    lst1=[len(dataset.train_positem_byuid[uid]) for uid in dataset.list_uid]
    lst2=[len(dataset.test_positem_byuid[uid]) for uid in dataset.list_uid]
    lst3=[lst1[idx]+lst2[idx] for idx,x in enumerate(lst1)]
    print(max(lst3),min(lst3),np.mean(lst3),np.std(lst3))

    lst1=[len(dataset.train_negitem_byuid[uid]) for uid in dataset.list_uid]
    lst2=[len(dataset.test_negitem_byuid[uid]) for uid in dataset.list_uid]
    lst3=[lst1[idx]+lst2[idx] for idx,x in enumerate(lst1)]
    print(max(lst3),min(lst3),np.mean(lst3),np.std(lst3))

    lst1=[len(dataset.train_posuser_byitemid[itemid]) for itemid in dataset.list_itemid]
    lst2=[len(dataset.test_posuser_byitemid[itemid]) for itemid in dataset.list_itemid]
    lst3=[lst1[idx]+lst2[idx] for idx,x in enumerate(lst1)]
    print(max(lst3),min(lst3),np.mean(lst3),np.std(lst3))

    lst1=[len(dataset.train_neguser_byitemid[itemid]) for itemid in dataset.list_itemid]
    lst2=[len(dataset.test_neguser_byitemid[itemid]) for itemid in dataset.list_itemid]
    lst3=[lst1[idx]+lst2[idx] for idx,x in enumerate(lst1)]
    print(max(lst3),min(lst3),np.mean(lst3),np.std(lst3))

basic_stat()


# In[7]:


class RecommendSysUtil():
    def set_distant(self, i, j):
        users_i = dataset.train_rateduser_byitemid[i]
        users_j = dataset.train_rateduser_byitemid[j]
        if (len(users_j) != 0):
            return 1 - 1.0 * len(set(users_i) & set(users_j)) / len(users_j)
        else:
            return 1.0

    def cosine_distant(self, i, j):
        vec_i = self.ratings_byitemid[i]
        vec_j = self.ratings_byitemid[j]
        return 1 - np.dot(vec_i, vec_j)

    def distant(self, i, j):
        if DISTANT_TYPE == 0:
            return self.set_distant(i, j)
        else:
            return self.cosine_distant(i, j)

    def novelty(self, uid, item_i):
        items_byu = self.dataset.train_rateditem_byuid[uid]
        if NOVELTY_TYPE == 0:
            return np.mean(
                [self.distant_mat[item_i][item_j] for item_j in items_byu])
        else:
            return -np.log2(
                len(self.dataset.train_rateduser_byitemid[item_i]
                    ) / len(self.dataset.list_uid) + pow(10, -9))

    def item_vectorize(self, itemid, is_dummy=0):
        vec = []
        data = self.dataset.df_iteminfo[self.dataset.df_iteminfo['itemid'] == (
            itemid)]
        #print(data)
        for attr in self.dataset.df_iteminfo.keys():
            value = list(data[attr])[0]
            if (type(value) == str):
                vec.append(int(value))
            else:
                vec.append(value)
        return vec

    def user_vectorize(self, uid, is_dummy=0):
        vec = []

        data = self.dataset.df_userinfo[self.dataset.df_userinfo['uid'] == (
            uid)]
        for attr in self.dataset.df_userinfo.keys():
            value = list(data[attr])[0]
            if (type(value) == str):
                vec.append(int(value))
            else:
                vec.append(value)
        return vec

    def df_onehot_encode(self, df):
        onehot_by_attr = {}
        for attr in df:
            if (type(df[attr][0]) == str):
                onehot_by_attr[attr] = max(df[attr].astype('int')) + 1
                #print(attr,onehot_by_attr[attr])
        return onehot_by_attr

    def save_vec(self):
        self.uid_to_vec = {}
        self.itemid_to_vec = {}
        sz = len(self.dataset.list_uid)
        for uid in self.dataset.list_uid:
            if uid % 1000 == 0:
                print("save user vec", uid, '/', sz)
            #print(self.user_vectorize(uid))
            self.uid_to_vec[uid] = self.user_vectorize(uid)
        sz = len(self.dataset.list_itemid)
        for itemid in self.dataset.list_itemid:
            if itemid % 1000 == 0:
                print("save item vec", itemid, '/', sz)
            self.itemid_to_vec[itemid] = self.item_vectorize(itemid)
          
    def save_distance(self):
        self.ratings_byitemid=[]
        for itemid in dataset.list_itemid:
            print('save item',itemid,'rating vector')
            vec=[0.0 for uid in self.dataset.list_uid]
            for uid in self.dataset.train_rateduser_byitemid[itemid]:
                #print(self.dataset.ratings_byitemid[itemid][uid])
                vec[uid]=self.dataset.ratings_byitemid[itemid][uid]
            vec=np.array(vec)+pow(10,-9)
            self.ratings_byitemid.append(vec/ np.linalg.norm(vec))
            
        self.distant_mat=[]
        for index_i,i in enumerate(self.dataset.list_itemid):
            print('save distance(%d/%d)'%(i,len(self.dataset.list_itemid)))
            self.distant_mat.append([])
            for index_j,j in enumerate(self.dataset.list_itemid):
                if index_j>index_i:
                    self.distant_mat[index_i].append(self.distant(index_i,index_j))
                elif index_j==index_i:
                    self.distant_mat[index_i].append(0)
                else:
                    self.distant_mat[index_i].append(self.distant_mat[index_j][index_i])
                
    def __init__(self, dataset):
        self.dataset = dataset
        self.useronehot_by_attr = self.df_onehot_encode(
            self.dataset.df_userinfo)
        self.itemonehot_by_attr = self.df_onehot_encode(
            self.dataset.df_iteminfo)
        self.save_vec()
        self.save_distance()


# In[8]:


try:
    f=open(UTILOBJ_PATH,"rb")
    util=pickle.load(f)
except:
    util=RecommendSysUtil(dataset)
    f=open(UTILOBJ_PATH,"wb")
    pickle.dump(util,f)


# In[17]:


class Nov_Distri_Saver():
    def __init__(self):
        self.pos_distr = {}
        self.neg_distr = {}


#   This class implements the algorithm in the paper.
#   1.First, it calculates novelty distribution and saves it in the hash (self.load_distribution())
#   2.Then, it trains the dataset by the alorithm in the paper (self.generate_a_batch(),self.train())
#
class RecommendSys():

    #   Get the novelty distribution of user u
    #   The data type of the distribution is the list of the novelty between user u and all the items
    #   noveltyb_list:
    #      list[novelty^β(user_u,item1),novelty^β(user_u,item2),....]
    #   noveltyreb_list:
    #      list[novelty^-β(user_u,item1),novelty^-β(user_u,item2),....]
    def get_novelty_distribution(self, u):
        list_positemid = self.dataset.train_positem_byuid[u]
        list_negitemid = self.dataset.train_negitem_byuid[u]
        positem_novdistr = [
            pow(self.util.novelty(u, itemid), self.beta)
            for itemid in list_positemid
        ]
        negitem_novdistr = [1.0 for itemid in list_negitemid]
        return positem_novdistr / np.sum(
            positem_novdistr), negitem_novdistr / np.sum(negitem_novdistr)

    def load_distribution(self):
        list_uid = self.dataset.list_uid
        tmp = Nov_Distri_Saver()
        for uid in list_uid:
            #print('load the novelty distribution of user', uid)
            tmp.pos_distr[uid], tmp.neg_distr[
                uid] = self.get_novelty_distribution(uid)
        return tmp

    def predict(self, list_uid, list_itemid):
        user_batch = [self.util.uid_to_vec[uid] for uid in list_uid]

        item_batch = []

        for itemid in list_itemid:
            item_batch.append(self.util.itemid_to_vec[itemid])

        label_batch = [[1] * len(list_itemid) for uid in list_uid]
        prob_matrix = self.prob.eval(
            feed_dict={
                self.user_input: user_batch,
                self.item_input: item_batch,
                self.label: label_batch
            })

        return prob_matrix

    def predict_by_queue(self, list_uid, list_itemid):
        sz = len(list_itemid)
        batch_sz = 10000
        bins = int(sz / batch_sz)
        ret = []
        for idx in range(bins):
            print('predict_by_queue %d/%d' % (idx, bins))
            tmp = self.predict(
                list_uid, list_itemid[idx * batch_sz:(idx + 1) * batch_sz])
            if ret != []:
                ret = np.concatenate((ret, tmp), axis=1)
            else:
                ret = tmp
        tmp = self.predict(list_uid, list_itemid[bins * batch_sz:])
        if ret != []:
            ret = np.concatenate((ret, tmp), axis=1)
        else:
            ret = tmp
        return ret

    def eval_performance(self):

        list_uid = dataset.list_uid
        list_itemid = dataset.list_itemid
        self.prob_by_uitem = self.predict_by_queue(list_uid, list_itemid)
        self.uid_to_recomm = self.base_recommend(self.prob_by_uitem,
                                                 self.top_N)
        #print(list_uid)
        #print(uid_to_recomm)
        acc = self.print_accuracy(self.uid_to_recomm, self.prob_by_uitem)
        reward0, reward1, agg_div, entro_div = self.print_diversity(
            self.uid_to_recomm)
        return reward0, reward1, agg_div, entro_div

    def print_accuracy(self, uid_to_recomm, prob_by_uitem):
        acc = 0
        for uid in self.dataset.list_uid:
            if len(self.dataset.test_positem_byuid[uid]) < self.top_N:
                continue
                #pass
            positem_test = list(self.dataset.test_positem_byuid[uid])

            if len(set(positem_test) & set(uid_to_recomm[uid])) != 0:
                acc += 1
        return acc / len(uid_to_recomm)

    def base_recommend(self, prob_by_uitem, top_N):
        uid_to_recomm = {}
        for uid in self.dataset.list_uid:
            if len(self.dataset.test_positem_byuid[uid]) < self.top_N:
                continue
                #pass
            prob_row = prob_by_uitem[uid]
            prob_arr = list(zip(self.dataset.list_itemid, prob_row))
            prob_arr = sorted(prob_arr, key=lambda d: -d[1])
            cnt = 0
            uid_to_recomm[uid] = []
            for pair in prob_arr:
                itemid = pair[0]
                if itemid not in dataset.train_rateditem_byuid[uid]:
                    uid_to_recomm[uid].append(itemid)
                    cnt += 1
                    if cnt == top_N:
                        break
        return uid_to_recomm

    def print_diversity(self, uid_to_recomm):
        avg_reward0 = 0.0
        avg_reward1 = 0.0
        agg_div = 0.0
        enp_div = 0.0

        cnt = 0
        for uid in uid_to_recomm:
            reward0 = 0.0
            reward1 = 0.0
            for itemid in uid_to_recomm[uid]:
                if (itemid in self.dataset.test_positem_byuid[uid]):
                    nov = self.util.novelty(uid, itemid)
                    if nov == np.inf or np == -np.inf:
                        nov = 0
                    if nov != 0:
                        nov0 = pow(nov, 0)
                        nov1 = pow(nov, 1)
                        cnt += 1
                    reward0 = max(reward0, nov0)
                    reward1 = max(reward1, nov1)
            avg_reward0 += reward0
            avg_reward1 += reward1

        if avg_reward0 != 0:
            avg_reward0 /= len(uid_to_recomm)
        if avg_reward1 != 0:
            avg_reward1 /= cnt

        recomm_set = set()
        cnt = 0
        self.rec_cnt[self.beta] = {i: 0 for i in dataset.list_itemid}
        for uid in uid_to_recomm:
            recomm_set = recomm_set | set(uid_to_recomm[uid])
            for i in uid_to_recomm[uid]:
                self.rec_cnt[self.beta][i] += 1
                cnt += 1
        agg_div = len(recomm_set) / len(uid_to_recomm) / self.top_N

        itemid_to_recomuser = {}

        for uid in uid_to_recomm:
            for itemid in uid_to_recomm[uid]:
                if itemid not in itemid_to_recomuser:
                    itemid_to_recomuser[itemid] = 0
                itemid_to_recomuser[itemid] += 1

        s = 0
        for itemid in itemid_to_recomuser:
            s += itemid_to_recomuser[itemid]

        for itemid in itemid_to_recomuser:
            probb = itemid_to_recomuser[itemid] / s + pow(10, -9)
            enp_div += -(np.log2(probb) * probb)

        #print('over diver %f'%(time.time()-t1))
        print(
            'Diversity: reward(β=0)=%.5f reward(β=1)=%.5f aggdiv=%.5f entropydiv=%.5f'
            % (avg_reward0, avg_reward1, agg_div, enp_div))
        return avg_reward0, avg_reward1, agg_div, enp_div

    def train_a_batch(self, iter, session):

        loss_all = 0

        user_batch = []
        item_batch = []
        label_batch = []
        list_positemid = []
        list_uid = []
        list_label = []
        list_negitemid = []

        for i in range(self.batch_size):
            uid = 0
            while (True):
                uid = random.randint(1, self.NUM_USERS)
                dataset = self.dataset
                if ((uid in dataset.list_uid)
                        and len(dataset.train_positem_byuid[uid]) != 0
                        and len(dataset.train_negitem_byuid[uid]) != 0):
                    break
            list_uid.append(uid)

        for uid in list_uid:
            pos_itemid = np.random.choice(
                self.dataset.train_positem_byuid[uid], p=self.pos_distr[uid])
            list_positemid.append(pos_itemid)
            list_label.append(1)
            user_batch.append(self.util.user_vectorize(uid))
            pos_itemvec = self.util.item_vectorize(pos_itemid)
            item_batch.append(pos_itemvec)

        prob_by_uitem = self.predict(list_uid, list_positemid)

        #print('predict end time '+time.asctime())

        #print('neg fetch start time '+time.asctime())

        neg_itemset = set()
        neg_index = {}
        for uid in list_uid:
            neg_itemset = neg_itemset | set(dataset.train_negitem_byuid[uid])
        for index, neg_item in enumerate(neg_itemset):
            neg_index[neg_item] = index
        neg_itemset = list(neg_itemset)
        neg_prob_by_uitem = self.predict(list_uid, neg_itemset)

        violator_cnt = 0
        for i, uid in enumerate(list_uid):
            neg_itemid = -1
            pos_itemid = list_positemid[i]
            pos_prob = prob_by_uitem[i][i]
            for k in range(self.LIMIT):
                neg_itemid = np.random.choice(
                    self.dataset.train_negitem_byuid[uid],
                    p=self.neg_distr[uid])
                neg_prob = neg_prob_by_uitem[i][neg_index[neg_itemid]]
                if neg_prob >= pos_prob and neg_prob != 0:
                    break
                else:
                    neg_itemid = -1

            if neg_itemid != -1:
                violator_cnt += 1
                list_label.append(-1)
                user_batch.append(self.util.user_vectorize(uid))
                neg_itemvec = self.util.item_vectorize(neg_itemid)
                item_batch.append(neg_itemvec)

        label_batch = [[1] * len(user_batch) for j in range(len(user_batch))]
        for i, label in enumerate(list_label):
            label_batch[i][i] = label

        feed_dict = {
            self.user_input: user_batch,
            self.item_input: item_batch,
            self.label: label_batch
        }
        if iter != 0:
            [_optimize, _loss] = session.run(
                [self.optimize, self.loss], feed_dict=feed_dict)
        else:
            [_loss] = session.run([self.loss], feed_dict=feed_dict)

        return _loss

    def read_distribution(self, nov_distri_path):
        try:
            f = open(nov_distri_path, "rb")
            tmp = pickle.load(f)
            self.neg_distr = tmp.neg_distr.copy()
            self.pos_distr = tmp.pos_distr.copy()
        except:
            tmp = self.load_distribution()
            f = open(nov_distri_path, "wb")
            pickle.dump(tmp, f)
            self.neg_distr = tmp.neg_distr.copy()
            self.pos_distr = tmp.pos_distr.copy()

    def cal_val_loss(self):
        p=[]
        q=[]
        prob_by_uitem = self.predict(self.dataset.list_uid, self.dataset.list_itemid)
        for uid in self.dataset.list_uid:
            for itemid in self.val_positem_byuid[uid]:
                prob=prob_by_uitem[uid][itemid]
                p.append(1)
                q.append(prob)
            for itemid in self.val_negitem_byuid[uid]:
                prob=prob_by_uitem[uid][itemid]
                p.append(0)
                q.append(prob)
        q=[x+1e-20 for x in q]
        return scipy.stats.entropy(p, q)

    def process_train(self, is_early_stopping):
        self.dataset=copy.deepcopy(dataset)
        
        if is_early_stopping==1:
            self.dataset.train_positem_byuid,self.val_positem_byuid=self.dataset.split_dict(self.dataset.train_positem_byuid,5/7)
            self.dataset.train_negitem_byuid,self.val_negitem_byuid=self.dataset.split_dict(self.dataset.train_negitem_byuid,5/7)
        
                
    def train(self,
              nov_distri_path,
              model_path,
              is_early_stopping=0,
              beta=0.0,
              batch_size=128,
              learning_rate=0.006,
              nu=0.0001,
              embedding_size=600,
              EVERY_N_ITERATIONS=100,
              MAX_ITERATIONS=0,
              predict_pair=[]):
        self.beta = beta
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.nu = nu
        self.embedding_size = embedding_size
        self.EVERY_N_ITERATIONS = EVERY_N_ITERATIONS
        self.MAX_ITERATIONS = MAX_ITERATIONS

        nov_distri_path = MODEL_DIR + nov_distri_path
        model_path = MODEL_DIR + model_path
        self.process_train(is_early_stopping)
        if is_early_stopping==1:
            nov_distri_path+="_es"
            model_path+="_es"
        self.read_distribution(nov_distri_path)
        # Create the TF graph
        graph = tf.Graph()
        dataset = self.util.dataset
        with graph.as_default(), tf.device('/cpu:0'):
            self.user_input = tf.placeholder(
                tf.int32, shape=[None, dataset._USER_SIZE], name='user_info')
            self.item_input = tf.placeholder(
                tf.int32, shape=[None, dataset._ITEM_SIZE], name='item_info')
            self.label = tf.placeholder(
                tf.int32, shape=[None, None], name='label')

            # Variables
            # embedding for users
            W = tf.Variable(
                initial_value=tf.truncated_normal(
                    (self.embedding_size, dataset._USER_SIZE_BIN),
                    stddev=1.0 / np.sqrt(self.embedding_size)))
            # embedding for movies
            A = tf.Variable(
                initial_value=tf.truncated_normal(
                    (self.embedding_size, dataset._ITEM_SIZE_BIN),
                    stddev=1.0 / np.sqrt(self.embedding_size)))
            # intercept
            b = tf.Variable(
                initial_value=tf.truncated_normal(
                    (self.embedding_size, 1),
                    stddev=1.0 / np.sqrt(self.embedding_size)))

            # select and sum the columns of W depending on the input
            w_offsets = [0] + [
                sum(dataset._USER_SIZE_OF_MASK_FIELDS[:i + 1])
                for i, j in enumerate(dataset._USER_SIZE_OF_MASK_FIELDS[:-1])
            ]
            w_offsets = tf.matmul(
                tf.ones(
                    shape=(tf.shape(self.user_input)[0], 1), dtype=tf.int32),
                tf.convert_to_tensor([w_offsets]))
            w_columns = self.user_input[:, :-dataset.
                                        _USER_SIZE_ONLY_NUM] + w_offsets  # last column is not an index
            w_selected = tf.gather(W, w_columns, axis=1)
            # age * corresponding column of W
            aux = tf.matmul(
                W[:, -dataset._USER_SIZE_ONLY_NUM:],
                tf.transpose(
                    tf.to_float(
                        (self.user_input[:, -dataset._USER_SIZE_ONLY_NUM:]))))
            batch_age = tf.reshape(
                aux,
                shape=(self.embedding_size, tf.shape(self.user_input)[0], 1))
            w_with_age = tf.concat([w_selected, batch_age], axis=2)
            w_result = tf.reduce_sum(w_with_age, axis=2)

            # select and sum the columns of A depending on the input
            a_offsets = [0] + [
                sum(dataset._ITEM_SIZE_OF_MASK_FIELDS[:i + 1])
                for i, j in enumerate(dataset._ITEM_SIZE_OF_MASK_FIELDS[:-1])
            ]
            a_offsets = tf.matmul(
                tf.ones(
                    shape=(tf.shape(self.item_input)[0], 1), dtype=tf.int32),
                tf.convert_to_tensor([a_offsets]))
            a_columns = self.item_input[:, :-dataset.
                                        _ITEM_SIZE_ONLY_NUM] + a_offsets  # last two columns are not indices
            a_selected = tf.gather(A, a_columns, axis=1)
            # dates * corresponding last two columns of A
            aux = tf.matmul(
                A[:, -dataset._ITEM_SIZE_ONLY_NUM:],
                tf.transpose(
                    tf.to_float(
                        self.item_input[:, -dataset._ITEM_SIZE_ONLY_NUM:])))
            batch_dates = tf.reshape(
                aux,
                shape=(self.embedding_size, tf.shape(self.item_input)[0], 1))
            # ... and the intercept
            intercept = tf.gather(
                b,
                tf.zeros(
                    shape=(tf.shape(self.item_input)[0], 1), dtype=tf.int32),
                axis=1)
            a_with_dates = tf.concat(
                [a_selected, batch_dates, intercept], axis=2)
            a_result = tf.reduce_sum(a_with_dates, axis=2)

            # Definition of g (Eq. (14) in the paper g = <Wu, Vi> = u^T * W^T * V * i)
            g = tf.matmul(tf.transpose(w_result), a_result)

            x = tf.to_float(self.label) * g
            self.prob = tf.nn.sigmoid(x)

            self.loss = tf.reduce_mean(tf.nn.softplus(tf.diag_part(-x)))

            # Regularization
            reg = self.nu * (tf.nn.l2_loss(W) + tf.nn.l2_loss(A))
            # Loss function with regularization (what we want to minimize)
            loss_to_minimize = self.loss + reg

            self.optimize = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(
                    loss=loss_to_minimize)
        # Once thep graph is created, let's probgram the training loop
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config,graph=graph) as session:
            # mandatory: initialize variables in the graph, i.e. W, A
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            try:
                saver.restore(session, model_path)
            except:
                #print()
                pass

            #result=[]
            self.EARLY_STOP_INTERVAL = 40
            average_loss = 0.0
            for iter in range(self.MAX_ITERATIONS + 1):
                train_loss = self.train_a_batch(iter, session)
                average_loss += train_loss
                print('Iteration', iter, 'Train_loss', train_loss)

                if iter % self.EVERY_N_ITERATIONS == 0:
                    reward0, reward1, agg_div, entro_div = self.eval_performance(
                    )
                if iter % self.EARLY_STOP_INTERVAL == 0:
                    average_loss = average_loss / self.EARLY_STOP_INTERVAL
                    print('Average_loss', average_loss)
                    if is_early_stopping==1:
                        stop_flg = self.early_stop(iter, self.cal_val_loss(), saver, session,
                                              model_path)
                        average_loss = 0.0
                        if stop_flg == 1:
                            break
                    else:
                        model_path = saver.save(session, model_path)
                        print('current model save in', model_path)
                        
            user_list = []
            item_list = []
            for (user, item) in predict_pair:
                user_list.append(user)
                item_list.append(item)
            if len(predict_pair) != 0:
                result = self.predict(user_list, item_list)
            else:
                result = {}
            return result, reward0, reward1, agg_div, entro_div
    
    def update_loss_win(self,loss):
        self.stop_loss = self.stop_loss[1:]
        self.stop_loss.append(loss)
        l1, l2, l3, l4, l5 = self.stop_loss
        return (l5 >= l4 and l5 >= l3 and l5 >= l2 and l5 >= l1) and (l4 >= l3 and l4 >= l2 and l4 >= l1)
    def long_not_improve(self,iter,loss):
        self.cur_iter=iter
        if loss<self.best_loss:
            self.best_loss=loss
            self.best_iter=iter
        return self.cur_iter-self.best_iter>=400
    def early_stop(self, iter, loss, saver, session, save_path):
        if iter == 0:
            self.stop_loss = [9999, 9999, 9999, 9999, 9999]
            self.best_loss=9999
            self.best_iter=0
            self.cur_iter=0
            return 0
        else:

            stop_flg = self.long_not_improve(iter,loss)#self.update_loss_win(loss)
            print({"loss":loss,"best_loss":self.best_loss,"best_iter":self.best_iter,"stop_win":self.stop_loss})
            if stop_flg==0:
                save_path = saver.save(session, save_path)
                print('current model save in', save_path)
            return stop_flg

    def __init__(self, util):
        self.rec_cnt = {}
        self.top_N = 10
        self.LIMIT = 100
        self.util = util
        self.dataset = util.dataset
        self.NUM_USERS = len(self.util.dataset.df_userinfo)
        self.NUM_ITEMS = len(self.util.dataset.df_iteminfo)
        self.neg_distr = {}
        self.pos_distr = {}
        self.beta = 0.0


# In[19]:


sys=RecommendSys(util)

beta_list=[0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
print(beta_list)
result_list=[]

for beta in list(beta_list):
    print(beta)
    s1="ml_nov_distri_beta%.1f"%(beta)
    s2="ml_K_600_beta_%.1f_vald2"%(beta)
#     result,reward0, reward1, agg_div, entro_div=sys.train(
#     s1,s2,beta=beta,is_early_stopping=1,predict_pair=[],MAX_ITERATIONS=4000)
#     with open("maxiter_%.2f"%(beta),"w") as f:
#         f.write(str(sys.cur_iter))
#     print('bestiter',sys.cur_iter)
    result,reward0, reward1, agg_div, entro_div=sys.train(
    s1,s2,beta=beta,is_early_stopping=0, predict_pair=[],MAX_ITERATIONS=0)
    result_list.append((beta,reward0,reward1,agg_div,entro_div))
pd.DataFrame(result_list,columns=["beta","avg_accuracy","avg_reward","agg_div","entropy_div"]).to_csv("mlens_newmethod_result_"+str(SEED)+".csv",index=False)


# In[21]:


s1="ml_nov_distri_beta%.1f"%(0.0)
s2="ml_K_600_beta_%.1f_vald2"%(0.0)
result,reward0, reward1, agg_div, entro_div=sys.train(
    s1,s2,beta=0.0,is_early_stopping=0, predict_pair=[],MAX_ITERATIONS=0)
def run_baseline():
    def find_best_fobj(uid, R, S, rel_matrix):
        fobj_set = []
        for index, i in enumerate(R):
            rel = rel_matrix[uid][i]
            min_dist = 1.0
            for j in S:
                dist = util.distant_mat[i][j]
                min_dist = min(dist, min_dist)
            fobj = (1 - k) * rel + k * min_dist
            #print(rel,min_dist)
            fobj_set.append((i, fobj))
        
        pair = max(fobj_set, key=lambda x: x[1])
        return pair[0]
    result_list = []
    k_list=[]
    if DISTANT_TYPE==0:
        k_list=[0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,
        0.09,0.1,1.0]
    else:
        k_list=[0.0,0.02,0.06,0.1,0.12,0.16,0.2,0.22,0.26,0.3,0.32,
         0.36,0.4,0.5,0.6,0.7,0.8]
    for k in k_list:
        print("lambda=%f" % (k))
        rel_matrix = sys.prob_by_uitem
        uid_to_recommend = sys.base_recommend(rel_matrix, 500).copy()
        for index1, uid in enumerate(uid_to_recommend):
            R, S = uid_to_recommend[uid], []
            #print(index1, '/', len(uid_to_recommend))
            for iter in range(sys.top_N):
                besti = find_best_fobj(uid, R, S, rel_matrix)
                R.remove(besti)
                S.append(besti)
            uid_to_recommend[uid] = S
        acc = sys.print_accuracy(uid_to_recommend, rel_matrix)
        print('Baseline Performance')
        avg_reward0, avg_reward1, agg_div, enp_div = sys.print_diversity(
            uid_to_recommend)
        result_list.append((k, avg_reward0, avg_reward1, agg_div, enp_div))
    pd.DataFrame(
         result_list,
         columns=[
             "lambda", "avg_accuracy", "avg_reward", "agg_div", "entropy_div"
         ]).to_csv(
             "mlens_baseline_result_" + str(SEED) + "_eq15.csv", index=False)


run_baseline()


# In[ ]:




