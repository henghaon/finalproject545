#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy.io as sio
import time 


# In[2]:


def g_1(x):
    return x+np.multiply(x>0.5,1-2*x)


# In[3]:


def g_2(x):
    x_norm=np.diag(np.sqrt(np.dot(np.transpose(x),x)))
    p=x_norm<=1
    return (1-x_norm)**6*(35*x_norm**2+18*x_norm+3)*p


# In[4]:


def Ker1(x,y):
    return 1+min(x,y)


# In[5]:


def Ker2(x,y):
    e=x-y
    e_norm=np.sqrt(np.dot(np.transpose(e),e))
    p=e_norm<=1
    return (1-e_norm)**4*(4*e_norm+1)*p


# In[6]:


def KRR_Ker1(X_train,y_train,X_test,y_test,k,lamb):
    n=y_train.shape[1]
    K1=np.zeros((n,n))
    K2=np.zeros((n,n))
    error=np.zeros((k,1))
    for i in range(n):
        for j in range(n):
            K1[i,j]=Ker1(X_train[:,i],X_train[:,j])
            K2[i,j]=Ker1(X_test[:,i],X_train[:,j])
    D=np.linalg.inv(lamb*n*np.identity(n)+K1)
    a_res=np.zeros((n,1))
    for i in range(k):
        a=np.dot(D,np.transpose(y_train-np.transpose(np.dot(K1,a_res))))
        a_res+=a
        prediction=np.transpose(np.dot(K2,a_res))
        e=prediction-y_test
        error[i,0]=np.dot(e,np.transpose(e))/n
    opt_k=np.argmin(error)
    return error[opt_k],opt_k,error


# In[166]:


D=2000
lamb=0.0002*np.array([2**x for x in range(11)])
k_1=150
k_2=300
m=len(lamb)
l=40
errors_1=np.zeros((m,l))
errors_2=np.zeros((m,l))
error_1=np.zeros((k_1,m))
error_2=np.zeros((k_2,m))
k1=np.zeros((m,l))
k2=np.zeros((m,l))


# In[167]:


for _ in range(l):
    x_1_train=np.random.uniform(low=0.0, high=1.0, size=D).reshape(1,D)
    epsilon_train=np.random.normal(0,0.2,D)
    y_1_train=g_1(x_1_train)+epsilon_train
    x_1_test=np.random.uniform(low=0.0, high=1.0, size=D).reshape(1,D)
    y_1_test=g_1(x_1_test)
    for i in range(m):
        errors_1[i][_],k1[i][_],error=KRR_Ker1(x_1_train,y_1_train,x_1_test,y_1_test,k_1,lamb[i])
        for q in range(k_1):
            error_1[q,i]+=error[q]/l
    print(_)
errors_1_ave=np.mean(errors_1,axis=1)
k1_ave=np.mean(k1,axis=1)


# In[168]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(7,6))
plt.xscale('log')
plt.yscale('log')
plt.plot(lamb,errors_1_ave,'r')
plt.xlabel(r'$\lambda$',fontsize=14)
plt.ylabel('EGE',fontsize=14)
plt.savefig('C:\\Files\\Assignments\\545\\final project\\ker1error.png')
_ = plt.tight_layout()


# In[169]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(7,6))
plt.xscale('log')
plt.plot(lamb,k1_ave,'b--')
plt.xlabel(r'$\lambda$',fontsize=14)
plt.ylabel('number of iteration',fontsize=14)
plt.savefig('C:\\Files\\Assignments\\545\\final project\\iter1.png')
_ = plt.tight_layout()


# In[170]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(7,6))
line1,=plt.plot(range(k_1),error_1[:,4],'r-',label=r'$\lambda=0.0032$')
line2,=plt.plot(range(k_1),error_1[:,6],'b--',label=r'$\lambda=0.0128$')
line3,=plt.plot(range(k_1),error_1[:,8],'m--',label=r'$\lambda=0.0512$')
line4,=plt.plot(range(k_1),error_1[:,10],'k.',label=r'$\lambda=0.2048$')
plt.xlabel('k',fontsize=14)
plt.ylabel('EGE',fontsize=14)
plt.legend(handles=[line1,line2,line3,line4])
plt.savefig('C:\\Files\\Assignments\\545\\final project\\ker1iter.png')
_ = plt.tight_layout()


# In[171]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(7,6))
plt.xscale('log')
line1,=plt.plot(lamb,error_1[0,:],'r-',label='k=1')
line2,=plt.plot(lamb,error_1[9,:],'b--',label='k=10')
line3,=plt.plot(lamb,error_1[19,:],'m--',label='k=20')
line4,=plt.plot(lamb,error_1[29,:],'k.',label='k=30')
line5,=plt.plot(lamb,error_1[39,:],'g-',label='k=40')
line6,=plt.plot(lamb,error_1[49,:],'y--',label='k=50')
plt.xlabel(r'$\lambda$',fontsize=14)
plt.ylabel('EGE',fontsize=14)
plt.legend(handles=[line1,line2,line3,line4,line5,line6])
plt.savefig('C:\\Files\\Assignments\\545\\final project\\ker1lamb.png')
_ = plt.tight_layout()


# In[132]:


def KRR_Ker2(X_train,y_train,X_test,y_test,k,lamb):
    n=y_train.shape[1]
    K1=np.zeros((n,n))
    K2=np.zeros((n,n))
    error=np.zeros((k,1))
    for i in range(n):
        for j in range(n):
            K1[i,j]=Ker2(X_train[:,i],X_train[:,j])
            K2[i,j]=Ker2(X_test[:,i],X_train[:,j])
    D=np.linalg.inv(lamb*n*np.identity(n)+K1)
    a_res=np.zeros((n,1))
    for i in range(k):
        a=np.dot(D,np.transpose(y_train-np.transpose(np.dot(K1,a_res))))
        a_res+=a
        prediction=np.transpose(np.dot(K2,a_res))
        e=prediction-y_test
        error[i,0]=np.dot(e,np.transpose(e))/n
    opt_k=np.argmin(error)
    return error[opt_k],opt_k,error


# In[134]:


for _ in range(l):
    x_2_train=np.random.rand(3,1,D).reshape(3,D)
    epsilon_train=np.random.normal(0,0.2,D)
    y_2_train=(g_2(x_2_train)+epsilon_train).reshape(1,D)
    x_2_test=np.random.rand(3,1,D).reshape(3,D)
    y_2_test=g_2(x_2_test)
    error=np.zeros((k_1,m))
    for i in range(m):
        errors_2[i][_],k2[i][_],error=KRR_Ker2(x_2_train,y_2_train,x_2_test,y_2_test,k_2,lamb[i])
        for q in range(k_2):
            error_2[q,i]+=error[q]/l
    print(_)
errors_2_ave=np.mean(errors_2,axis=1)
k2_ave=np.mean(k2,axis=1)


# In[135]:


def krr_asr(X_train,y_train,X_test,y_test,lamb,theta):
    n=y_train.shape[1]
    K1=np.zeros((n,n))
    K2=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K1[i,j]=Ker2(X_train[:,i],X_train[:,j])
            K2[i,j]=Ker2(X_test[:,i],X_train[:,j])
    D=np.linalg.inv(lamb*n*np.identity(n)+K1)
    a_res=np.zeros((n,1))
    N_lamb=np.trace(np.dot(D,K1))
    epsilon=theta*np.sqrt(lamb)/np.sqrt(n)*(1+(np.sqrt(n*lamb)+1)*np.sqrt(max(1,N_lamb))/n/lamb)*(np.sqrt(n*lamb)+1)*np.sqrt(max(1,N_lamb))/np.sqrt(n*lamb)
    k=0
    err=epsilon+1
    while err>epsilon:
        a=np.dot(D,np.transpose(y_train-np.transpose(np.dot(K1,a_res))))
        a_res+=a
        pred_train=np.transpose(np.dot(K1,a_res))
        e_train=pred_train-y_train
        err=np.sqrt(np.dot(np.dot(e_train,K1),np.transpose(e_train)))/n
        k+=1
    prediction=np.transpose(np.dot(K2,a_res))
    e=prediction-y_test
    error=np.dot(e,np.transpose(e))/n
    return k,error


# In[136]:


def krr_ora(X_train,y_train,X_test,y_test,lamb,k):
    n=y_train.shape[1]
    K1=np.zeros((n,n))
    K2=np.zeros((n,n))
    error=np.zeros((k,1))
    for i in range(n):
        for j in range(n):
            K1[i,j]=Ker1(X_train[:,i],X_train[:,j])
            K2[i,j]=Ker1(X_test[:,i],X_train[:,j])
    D=np.linalg.inv(lamb*n*np.identity(n)+K1)
    a_res=np.zeros((n,1))
    for i in range(k):
        a=np.dot(D,np.transpose(y_train-np.transpose(np.dot(K1,a_res))))
        a_res+=a
        prediction=np.transpose(np.dot(K2,a_res))
        e=prediction-y_test
        error[i,0]=np.dot(e,np.transpose(e))/n
    opt_k=np.argmin(error)
    return opt_k,error[opt_k]


# In[140]:


D=[800,1200,1600,2000,2400,2800,3200,3600,4000]
lamb=[0.016,0.032,0.064,0.128]
n_1=len(D)
n_2=len(lamb)
error_as=np.zeros((n_1,n_2))
error_or=np.zeros((n_1,n_2))
error_cv=np.zeros((n_1,n_2))
theta=0.05
k_1=150
for i in range(n_1):
    for j in range(n_2):
        d=D[i]
        x_train=np.random.uniform(low=0.0, high=1.0, size=d).reshape(1,d)
        epsilon_train=np.random.normal(0,0.2,d)
        y_train=g_1(x_train)+epsilon_train
        x_test=np.random.uniform(low=0.0, high=1.0, size=d).reshape(1,d)
        y_test=g_1(x_test)
        k,error_as[i,j]=krr_asr(x_train,y_train,x_test,y_test,lamb[j],theta)
        k,error_or[i,j]=krr_ora(x_train,y_train,x_test,y_test,lamb[j],k_1)
        k,error_cv[i,j]=krr_cv(x_train,y_train,x_test,y_test,lamb[j],k_1,5)
        print((i,j))


# In[109]:


def krr_cv(X_train,y_train,X_test,y_test,lamb,k_1,k_folds):
    n=y_train.shape[1]
    N=np.int(n/k_folds)
    n_train=n-N
    cv_errors=np.zeros((k_1,5))
    for i in range(k_folds):
        x_vali=X_train[:,N*i:N*(i+1)]
        x_trai=X_train[:,list(range(0,N*i))+list(range(N*(i+1),n))]
        y_vali=y_train[:,N*i:N*(i+1)]
        y_trai=y_train[:,list(range(0,N*i))+list(range(N*(i+1),n))]
        K1=np.zeros((n_train,n_train))
        K2=np.zeros((N,n_train))
        for l in range(n_train):
            for m in range(n_train):
                K1[l,m]=Ker2(x_trai[:,l],x_trai[:,m])
        for l in range(n_train):
            for m in range(N):
                K2[m,l]=Ker2(x_vali[:,m],x_trai[:,l])
        D=np.linalg.inv(lamb*n_train*np.identity(n_train)+K1)
        a_res=np.zeros((n_train,1))
        for k in range(k_1):
            a=np.dot(D,np.transpose(y_trai-np.transpose(np.dot(K1,a_res))))
            a_res+=a
            prediction=np.transpose(np.dot(K2,a_res))
            e=prediction-y_vali
            cv_errors[k,i]=np.dot(e,np.transpose(e))/n_train
    cv_error=np.mean(cv_errors,axis=1)
    opt_k=np.argmin(cv_error)
    K1=np.zeros((n,n))
    K2=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K1[i,j]=Ker1(X_train[:,i],X_train[:,j])
            K2[i,j]=Ker1(X_test[:,i],X_train[:,j])
    D=np.linalg.inv(lamb*n*np.identity(n)+K1)
    a_res=np.zeros((n,1))
    for i in range(opt_k):
        a=np.dot(D,np.transpose(y_train-np.transpose(np.dot(K1,a_res))))
        a_res+=a
    prediction=np.transpose(np.dot(K2,a_res))
    e=prediction-y_test
    error=np.dot(e,np.transpose(e))/n
    return opt_k,error


# In[179]:


get_ipython().run_line_magic('matplotlib', 'inline')
line1,=plt.plot(D,error_or[:,0],'b--',label='Oracle')
line2,=plt.plot(D,error_as[:,0],'r-',label='ASR')
line3,=plt.plot(D,error_cv[:,0],'k-',label='CV')
plt.legend(handles=[line1,line2,line3])
plt.xlabel('number of training examples',fontsize=14)
plt.ylabel('number of iteration',fontsize=14)
plt.savefig('C:\\Files\\Assignments\\545\\final project\\modelselect1.png')
_ = plt.tight_layout()


# In[180]:


get_ipython().run_line_magic('matplotlib', 'inline')
line1,=plt.plot(D,error_or[:,1],'b--',label='Oracle')
line2,=plt.plot(D,error_as[:,1],'r-',label='ASR')
line3,=plt.plot(D,error_cv[:,1],'k-',label='CV')
plt.legend(handles=[line1,line2,line3])
plt.xlabel('number of training examples',fontsize=14)
plt.ylabel('number of iteration',fontsize=14)
plt.savefig('C:\\Files\\Assignments\\545\\final project\\modelselect2.png')
_ = plt.tight_layout()


# In[181]:


get_ipython().run_line_magic('matplotlib', 'inline')
line1,=plt.plot(D,error_or[:,2],'b--',label='Oracle')
line2,=plt.plot(D,error_as[:,2],'r-',label='ASR')
line3,=plt.plot(D,error_cv[:,2],'k-',label='CV')
plt.legend(handles=[line1,line2,line3])
plt.xlabel('number of training examples',fontsize=14)
plt.ylabel('number of iteration',fontsize=14)
plt.savefig('C:\\Files\\Assignments\\545\\final project\\modelselect3.png')
_ = plt.tight_layout()


# In[182]:


get_ipython().run_line_magic('matplotlib', 'inline')
line1,=plt.plot(D,error_or[:,3],'b--',label='Oracle')
line2,=plt.plot(D,error_as[:,3],'r-',label='ASR')
line3,=plt.plot(D,error_cv[:,3],'k-',label='CV')
plt.legend(handles=[line1,line2,line3])
plt.xlabel('number of training examples',fontsize=14)
plt.ylabel('number of iteration',fontsize=14)
plt.savefig('C:\\Files\\Assignments\\545\\final project\\modelselect4.png')
_ = plt.tight_layout()


# In[ ]:




