import numpy as np
import random
from utils import *
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

    def __init__(self, movielens, split_ratio, seed=int(read_config('basic-settings','seed'))):
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
    

if __name__ == '__main__':
    from movielens_feature import MovieLens
    movielens=MovieLens()
    dataset=DataSetProcesser(movielens,0.7)
