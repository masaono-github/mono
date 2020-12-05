# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 00:29:57 2020

@author: shunsuke
"""

#データ行がサンプル、列が変数
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()


def clustering(X,K,m):
    """
    X:データセット
    K:設定したクラスター数
    m:設定したファジー度
    
    """
    X=np.array(X)
    #データ数
    N=X.shape[0]
    #データの次元
    D=X.shape[1]
    #初期値
    c=np.random.random([K,D])
    old_c=np.array([])
    #帰属度
    U=[]  
    kurikaesi=0
    
    while True:
        #帰属度の更新
        for x in X:
            u=[(1/np.linalg.norm(x-c[kk]))**(2/(m-1))/np.sum([(1/np.linalg.norm(x-c[k]))**(2/(m-1)) for k in range(K)]) for kk in range(K)]
            #u=np.array(u)
            U.append(u)
        #U=np.array(U)

        #クラスター中心の更新
        Um=[np.sum([U[i][k]**m for i in range(N)]) for k in range(K)]

        Um_x0=np.array([[(U[i][k]**m)*np.array(X[i]) for i in range(N)] for k in range(K)])
        Um_x=np.array([sum(Um_x0[k]) for k in range(K)])          
        old_c=c
        c=np.array([Um_x[k]/Um[k] for k in range(K)])
        
        #収束条件
        z=0
        for k in range(K):
            if z<max(np.abs(c[k]-old_c[k])):
                z=max(np.abs(c[k]-old_c[k]))
        if z<=0.00001:
            print('収束条件を満たしたため、終了')
            break
        
        #反復回数の指定
        kurikaesi+=1
        if kurikaesi>=100:
            print('{}回繰り返し後、終了\n'.format(kurikaesi))
            break
        U=[]
    return c,U


#クラスター数
K=3
#Fuzzyパラメータ
m=2.0

#クラスタリング
mu=clustering(iris.data,K,m)

#帰属クラスター結果
cluster_num=[]
for i in mu[1]:
    for j in range(len(i)):
        if i[j]==max(i):
            cluster_num.append(j+1)
        

print('クラスター重心:\n{}\n'.format(mu[0]))
print('帰属クラスター:\n{}\n'.format(cluster_num))
