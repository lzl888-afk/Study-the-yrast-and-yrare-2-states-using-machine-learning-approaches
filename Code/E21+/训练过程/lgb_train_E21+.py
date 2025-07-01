import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

params = {'num_leaves': 10,
          'min_data_in_leaf': 30,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.001,
          "min_sum_hessian_in_leaf": 3,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.7,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          # "lambda_l2": 1.385029,
          "verbosity": -1,
          "nthread": 4,
          'metric': 'mse',
          "random_state": 2019,}
# %%
path = r'H:\科研1\E02E03w\激发态2+\机器学习数据/'
data1 = pd.read_csv(path + '机器学习数据训练.txt', sep='\s+', header=None).to_numpy()
data2 = pd.read_csv(path + '机器学习数据合并.txt', sep='\s+', header=None).to_numpy()
data3 = pd.read_csv(path + '机器学习数据外推.txt', sep='\s+', header=None).to_numpy()
# def index(data,Z,K,col,col1):
#     return [x for (x,m) in enumerate(np.column_stack((data[:,col],data[:,col1]))) if m[0] >= Z and m[1]>=K]
data1=pd.DataFrame(data1)
data2=pd.DataFrame(data2)
data3=pd.DataFrame(data3)
# data1[20]=data1[20].fillna(0)
#%%

data1.columns = ["Z","N",'A','Sp','Sn','S2p','S2n',"Saer","beata1","beata2","B","Blqu","B-Blqu","P","NT","PT","ZD","ND","E21","E41","E02","exp",'err']  # 相当于改列名
data2.columns = ["Z","N",'A','Sp','Sn','S2p','S2n',"Saer","beata1","beata2","B","Blqu","B-Blqu","P","NT","PT","ZD","ND","E21","E41","E02","exp",'err']  # 相当于改列名
data3.columns = ["Z","N",'A','Sp','Sn','S2p','S2n',"Saer","beata1","beata2","B","Blqu","B-Blqu","P","NT","PT","ZD","ND","E21","E41","E02","exp",'err']  # 相当于改列名





feats1 = ["Z","N",'A','Sp','Sn','S2p','S2n',"Saer","beata1","beata2","B","Blqu","B-Blqu","P","NT","PT"]

feats=[feats1]





# %%
loop=500
#percentage=[0.2,0.5,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.97,0.99]


percentage=[len(feats1)]



for j in range(len(feats)):

    x = np.array(data1[feats[j]])
    y = np.log10(np.array(data1["exp"]))
    X_he = np.array(data2[feats[j]])
    Y_he = np.log10(np.array(data2["exp"]))
    X_wai = np.array(data3[feats[j]])
    Y_wai = np.log10(np.array(data3["exp"]))

    for i in range(loop):

        X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.2,
                                                            random_state=random.randint(1, loop))
        # X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X_train, Y_train, test_size=0.1,
        #                                                     random_state=100)

        print(max(Y_val))
        print(min(Y_val))
        lgb_train = lgb.Dataset(X_train, Y_train)
        lgb_eval = lgb.Dataset(x, y, reference=lgb_train)
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=50000,
                        valid_sets=lgb_eval,
                        verbose_eval=100,
                        early_stopping_rounds=200)

        Y_pred_val = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        Y_pred_train = gbm.predict(X_train, num_iteration=gbm.best_iteration)
        Y_pred_he = gbm.predict(X_he, num_iteration=gbm.best_iteration)
        Y_pred_wai = gbm.predict(X_wai, num_iteration=gbm.best_iteration)

        ame_val = mean_absolute_error(Y_val, Y_pred_val)
        ame_train = mean_absolute_error(Y_train, Y_pred_train)
        ame_he = mean_absolute_error(Y_he, Y_pred_he)
        ame_wai = mean_absolute_error(Y_wai, Y_wai)

        rms_val = mean_squared_error(Y_val, Y_pred_val)**0.5
        rms_train = mean_squared_error(Y_train, Y_pred_train)**0.5
        rms_he = mean_squared_error(Y_he, Y_pred_he)**0.5
        rms_wai = mean_squared_error(Y_wai, Y_wai)**0.5

        if i == 0:
            Mod2_Y_pred_val = Y_pred_val
            Mod2_Y_pred_train = Y_pred_train
            Mod2_Y_pred_he = Y_pred_he
            Mod2_Y_pred_wai = Y_pred_wai

            Mod2_ame_val = ame_val
            Mod2_ame_train = ame_train
            Mod2_ame_he = ame_he
            Mod2_ame_wai = ame_wai

            Mod2_rms_val = rms_val
            Mod2_rms_train = rms_train
            Mod2_rms_he = rms_he
            Mod2_rms_wai = rms_wai

        else:
            Mod2_Y_pred_val = np.column_stack((Mod2_Y_pred_val, Y_pred_val))
            Mod2_Y_pred_train = np.column_stack((Mod2_Y_pred_train, Y_pred_train))
            Mod2_Y_pred_he = np.column_stack((Mod2_Y_pred_he, Y_pred_he))
            Mod2_Y_pred_wai = np.column_stack((Mod2_Y_pred_wai, Y_pred_wai))

            Mod2_ame_val = np.column_stack((Mod2_ame_val, ame_val))
            Mod2_ame_train = np.column_stack((Mod2_ame_train, ame_train))
            Mod2_ame_he = np.column_stack((Mod2_ame_he, ame_he))
            Mod2_ame_wai = np.column_stack((Mod2_ame_wai, ame_wai))

            Mod2_rms_val = np.column_stack((Mod2_rms_val, rms_val))
            Mod2_rms_train = np.column_stack((Mod2_rms_train, rms_train))
            Mod2_rms_he = np.column_stack((Mod2_rms_he, rms_he))
            Mod2_rms_wai = np.column_stack((Mod2_rms_wai, rms_wai))


        print('The mean_absolute_error of prediction is:Train', ame_train)
        print('The mean_squared_error of prediction is:Train', rms_train)

        print('The mean_absolute_error of prediction is:He', ame_he)
        print('The mean_squared_error of prediction is:He', rms_he)

        print('The mean_absolute_error of prediction is:Val', ame_val)
        print('The mean_squared_error of prediction is:val', rms_val)
        print("aaaaaaa", x.shape)
    pathh1 = r"H:\科研1\E02E03w\激发态2+\rms随比例变化\合并数据/"
    pathh2 = r"H:\科研1\E02E03w\激发态2+\rms随比例变化\训练数据/"
    pathh3 = r"H:\科研1\E02E03w\激发态2+\rms随比例变化\剩余数据/"
    pathh4 = r"H:\科研1\E02E03w\激发态2+\rms随比例变化\外推数据/"

    np.savetxt(pathh1 + 'pre_80_F' + "16" + '.txt',
               np.column_stack((Mod2_Y_pred_he, np.mean(Mod2_Y_pred_he, axis=1), np.std(Mod2_Y_pred_he, axis=1))), fmt='%10.8f  ')
    np.savetxt(pathh1 + 'ame_80_F' + "16" + '.txt',
               np.column_stack((Mod2_ame_he, np.mean(Mod2_ame_he, axis=1), np.std(Mod2_ame_he, axis=1))), fmt='%10.8f  ')
    np.savetxt(pathh1 + 'rms_80_F' + "16" + '.txt',
               np.column_stack((Mod2_rms_he, np.mean(Mod2_rms_he, axis=1), np.std(Mod2_rms_he, axis=1))), fmt='%10.8f  ')

    np.savetxt(pathh2 + 'pre_80_F' + "16" + '.txt',
               np.column_stack((Mod2_Y_pred_train, np.mean(Mod2_Y_pred_train, axis=1), np.std(Mod2_Y_pred_train, axis=1))), fmt='%10.8f  ')
    np.savetxt(pathh2 + 'ame_80_F' + "16" + '.txt',
               np.column_stack((Mod2_ame_train, np.mean(Mod2_ame_train, axis=1), np.std(Mod2_ame_train, axis=1))), fmt='%10.8f  ')
    np.savetxt(pathh2 + 'rms_80_F' + "16" + '.txt',
               np.column_stack((Mod2_rms_train, np.mean(Mod2_rms_train, axis=1), np.std(Mod2_rms_train, axis=1))), fmt='%10.8f  ')

    np.savetxt(pathh3 + 'pre_80_F' + "16" + '.txt',
               np.column_stack((Mod2_Y_pred_val, np.mean(Mod2_Y_pred_val, axis=1), np.std(Mod2_Y_pred_val, axis=1))),
               fmt='%10.8f  ')
    np.savetxt(pathh3 + 'ame_80_F' + "16" + '.txt',
               np.column_stack((Mod2_ame_val, np.mean(Mod2_ame_val, axis=1), np.std(Mod2_ame_val, axis=1))),
               fmt='%10.8f  ')
    np.savetxt(pathh3 + 'rms_80_F' + "16" + '.txt',
               np.column_stack((Mod2_rms_val, np.mean(Mod2_rms_val, axis=1), np.std(Mod2_rms_val, axis=1))),
               fmt='%10.8f  ')

    np.savetxt(pathh4 + 'pre_80_F' + "16" + '.txt',
               np.column_stack(
                   (Mod2_Y_pred_wai, np.mean(Mod2_Y_pred_wai, axis=1), np.std(Mod2_Y_pred_wai, axis=1))),
               fmt='%10.8f  ')
    np.savetxt(pathh4 + 'ame_80_F' + "16" + '.txt',
               np.column_stack((Mod2_ame_wai, np.mean(Mod2_ame_wai, axis=1), np.std(Mod2_ame_wai, axis=1))),
               fmt='%10.8f  ')
    np.savetxt(pathh4 + 'rms_80_F' + "16" + '.txt',
               np.column_stack((Mod2_rms_wai, np.mean(Mod2_rms_wai, axis=1), np.std(Mod2_rms_wai, axis=1))),
               fmt='%10.8f  ')

#%%
