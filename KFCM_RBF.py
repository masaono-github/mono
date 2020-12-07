# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 02:52:43 2020

@author: shunsuke
"""

#データ行がサンプル、列が変数
import numpy as np
import math

def rbf_kernel(x,y,a,b,sigma):#a=1,b=2の場合、通常のガウスカーネル関数
    x=np.array(x)
    y=np.array(y)
    ker_f=math.e**(-((np.linalg.norm(x**a-y**a))**b)/sigma**2)
    return ker_f
    

def KFCM(X,K,m):
    """
    X: 特徴ベクトルの集合
    K: クラスタ数
    m: Fuzzyパラメータ
    """
    #データ数
    N=X.shape[0]
    #データの次元
    D=X.shape[1]
    #初期値
    c=np.random.random([K,D])
    old_c=np.array([])
    #反復回数
    kurikaesi=0
    #RBFカーネル関数のパラメータ
    a=1
    b=1
    sigma=1.5

    while True:
        #帰属度の更新
        rbf=[]
        for x in X:
            rbf_x=[rbf_kernel(x,c[k],a,b,sigma) for k in range(K)]
            rbf.append(rbf_x)
        
        U=[]
        count=0
        for x in X:
            u_bunbo=np.sum([(1/(1-rbf[count][k]))**(1/(m-1)) for k in range(K)])
            u_bunsi=[(1/(1-rbf[count][k]))**(1/(m-1)) for k in range(K)]
            u=u_bunsi/u_bunbo
            U.append(u)
            count+=1

        #クラスター中心の更新
        Um=[np.sum([(U[i][k]**m)*rbf[i][k] for i in range(N)]) for k in range(K)]

        Um_x0=np.array([[(U[i][k]**m)*rbf[i][k]*np.array(X[i]) for i in range(N)] for k in range(K)])
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
              
        #上限反復回数の指定
        kurikaesi+=1
        if kurikaesi>=100:
            print('{}回繰り返し後、終了\n'.format(kurikaesi))
            break
        
    return c,U



###ここから実際にクラスタリング###
K=3
m=2.0

from sklearn.datasets import load_iris
iris = load_iris()
result=KFCM(iris.data,K,m)

#帰属クラスター結果
cluster_num=[]
for i in result[1]:
    for j in range(len(i)):
        if i[j]==max(i):
            cluster_num.append(j+1)
        
print('クラスター重心:\n{}\n'.format(result[0]))
#print('帰属クラスター:\n{}\n'.format(cluster_num))