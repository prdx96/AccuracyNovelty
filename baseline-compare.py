def run_baseline(sys,dis_type):
    
    s1="%s_nov_distri_beta%.1f"%(pref,0.0)
    s2="%s_K_600_beta_%.1f_vald2"%(pref,0.0)
    result,reward0, reward1, agg_div, entro_div=sys.train(
        s1,s2,beta=0.0,is_early_stopping=0, predict_pair=[],MAX_ITERATIONS=0)
        
        UTILOBJ_PATH='yahoo_util_%d_dis_%d.pkl'%(SEED,dis_type)
        try:
            f=open(UTILOBJ_PATH,"rb")
            util_tmp=pickle.load(f)
        except:
            util_tmp=RecommendSysUtil(dataset)
            f=open(UTILOBJ_PATH,"wb")
            pickle.dump(util_tmp,f)
        
        distant_mat=copy.deepcopy(util_tmp.distant_mat)
        def find_best_fobj(uid, R, S, rel_matrix):
            fobj_set = []
            for index, i in enumerate(R):
                rel = rel_matrix[uid][i]
                min_dist = 1.0
                for j in S:
                    dist = distant_mat[i][j]
                    min_dist = min(dist, min_dist)
                fobj = (1 - k) * rel + k * min_dist
                #print(rel,min_dist)
                fobj_set.append((i, fobj))
            
            pair = max(fobj_set, key=lambda x: x[1])
            return pair[0]
        result_list = []
        k_list=[]
        if dis_type==0:
            k_list=[0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,
            0.09,0.1,0.2,0.4,0.6,0.8]
        else:
            k_list=[0.0,0.02,0.06,0.1,0.12,0.16,0.2,0.22,0.26,0.3]
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
                "%s_baseline_result_"%(pref) + str(SEED) + "_eq%d.csv"%(14 if dis_type==1 else 15), index=False)