import scipy 
import pandas as pd
import numpy as np
from utils import *

class RecommendSysUtil():
    def set_distant(self, i, j):
        users_i = self.dataset.train_rateduser_byitemid[i]
        users_j = self.dataset.train_rateduser_byitemid[j]
        if (len(users_j) != 0):
            return 1 - 1.0 * len(set(users_i) & set(users_j)) / len(users_j)
        else:
            return 1.0

    def cosine_distant(self, i, j):
        vec_i = self.ratings_byitemid[i]
        vec_j = self.ratings_byitemid[j]
        return 1 - np.dot(vec_i, vec_j)

    def distant(self, i, j):
        if self.DISTANT_TYPE == 0:
            return self.set_distant(i, j)
        else:
            return self.cosine_distant(i, j)

    def novelty(self, uid, item_i):
        items_byu = self.dataset.train_rateditem_byuid[uid]
        if self.NOVELTY_TYPE == 0:
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
        for itemid in self.dataset.list_itemid:
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
        self.DISTANT_TYPE=int(read_config('novelty-settings','distant_type'))
        self.NOVELTY_TYPE=int(read_config('novelty-settings','novelty_type'))
        self.dataset = dataset
        self.useronehot_by_attr = self.df_onehot_encode(
            self.dataset.df_userinfo)
        self.itemonehot_by_attr = self.df_onehot_encode(
            self.dataset.df_iteminfo)
        self.save_vec()
        self.save_distance()

if __name__ == '__main__':
    from movielens_feature import MovieLens
    from accuracy_novelty_preprocessor import DataSetProcesser
    movielens=MovieLens()
    dataset=DataSetProcesser(movielens,0.7)
    util=RecommendSysUtil(dataset)


