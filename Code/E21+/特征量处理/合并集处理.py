import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split


"结合能数据处理"
path = r'H:\科研1\E02E03w/'
path1 = r'H:\科研1\E02E03w\激发态2+\机器学习数据/'
data = pd.read_csv(path1 + '合并数据.txt', sep='\s+', header=0).to_numpy()
AME2020=pd.read_csv(path+'包含评价结合能对照.txt',sep='\s+',header=0).to_numpy()
Beata1=pd.read_csv(path+'FRDM.txt',sep='\s+',header=0).to_numpy()
Beata2=pd.read_csv(path+'四级形变参数对照.txt',sep='\s+',header=0).to_numpy()
# E2=pd.read_csv(path+'E21.txt',sep='\s+',header=0).to_numpy()
# E4=pd.read_csv(path+'E41.txt',sep='\s+',header=0).to_numpy()
Z=data[:,0]
N=data[:,1]
A=data[:,2]
Rexp=data[:,3]
Exp_error=data[:,4]

e2=pd.read_csv(path+'E21.txt',sep='\s+',header=0).to_numpy()
e4=pd.read_csv(path+'E41.txt',sep='\s+',header=0).to_numpy()
e02=pd.read_csv(path+'E02.txt',sep='\s+',header=0).to_numpy()
E2=e2[:,:5]
E4=e4[:,:5]
E002=e02[:,:5]


#%%




B=np.zeros(len(data))
Sp=np.zeros(len(data))
S2p=np.zeros(len(data))
Sn=np.zeros(len(data))
S2n=np.zeros(len(data))
Saer=np.zeros(len(data))
Blqu=np.zeros(len(data))
beata1=np.zeros(len(data))
beata2=np.zeros(len(data))
E41=np.zeros(len(data))
E21=np.zeros(len(data))
E02=np.zeros(len(data))
"液滴结合能"
av = -15.4963
sa = 17.7937
kv = -1.8232
ks = -2.2593
ac = 0.7093
fp = -1.2739
dn = 4.6919
dp = 4.7230
dnp = -6.4920
a = Z + N
tz = abs((Z - N) / 2)
for i in range(len(data)):
    av = -15.4963
    sa = 17.7937
    kv = -1.8232
    ks = -2.2593
    ac = 0.7093
    fp = -1.2739
    dn = 4.6919
    dp = 4.7230
    dnp = -6.4920
    a = Z[i] + N[i]
    tz = abs((Z[i] - N[i]) / 2)
    if Z[i] % 2 == 0 and N[i] % 2 == 0:
        ep = 0
    if Z[i]% 2 == 0 and N[i]% 2 == 1:
        ep = dn / N[i] ** (1 / 3)
    if Z[i] % 2 == 1 and N[i]% 2 == 0:
        ep = dp / Z[i] ** (1 / 3)
    if Z[i] % 2 == 1 and N[i]% 2 == 1:
        ep = dn / N[i] ** (1 / 3) + dp / Z[i] ** (1 / 3) + dnp / a ** (2 / 3)
    Blqu[i] = -(av * (1 + 4 * kv * tz * (tz + 1) / a ** 2) * a
                 + sa * (1 + 4 * ks * tz * (tz + 1) / a ** 2) * a ** (2 / 3)
                 + ac * Z[i] ** 2 / a ** (1 / 3) + fp * Z[i] ** 2 / a + ep)
# Blqu=B(Z,N)
"实验结合能"
for i in range(len(data[:,0])):
    for j in range(len(AME2020)):
        if data[i, 0] == AME2020[j, 0] and data[i, 1] == AME2020[j, 1]:
            B[i] = AME2020[j, 3]

alldata=np.column_stack((data[:,0:3],B))
"分离能数据处理"
AME2020=pd.read_csv(path+'包含评价结合能对照.txt',sep='\s+',header=0).to_numpy()
# a=np.zeros(len(data))
for i in range(len(alldata)):
    for j in range(len(AME2020)):
        if alldata[i, 0] == AME2020[j, 0] + 1 and alldata[i, 1] == AME2020[j, 1]:
            Sp[i] = alldata[i, 3] - AME2020[j, 3]
        if alldata[i, 0] == AME2020[j, 0] + 2 and alldata[i, 1] == AME2020[j, 1]:
            S2p[i] = alldata[i, 3] - AME2020[j, 3]
        if alldata[i, 0] == AME2020[j, 0] and alldata[i, 1] == AME2020[j, 1] + 1:
            Sn[i] = alldata[i, 3] - AME2020[j, 3]
        if alldata[i, 0] == AME2020[j, 0] and alldata[i, 1] == AME2020[j, 1] + 2:
            S2n[i] = alldata[i, 3] - AME2020[j, 3]
        if  alldata[i, 0] == AME2020[j, 0]+2 and alldata[i, 1] == AME2020[j, 1] + 2:
            Saer[i] = alldata[i, 3] - AME2020[j, 3]-28.29566240
"四级形变参数处理"
for i in range(len(data)):
    for j in range(len(Beata1)):
        if data[i,0]==Beata1[j,0] and data[i,1]==Beata1[j,1]:
           beata1[i]=Beata1[j,3]
for i in range(len(data)):
    for j in range(len(Beata2)):
        if data[i,0]==Beata2[j,0] and data[i,1]==Beata2[j,1]:
           beata2[i]=Beata2[j,3]
r1=1.2269*A**(1/3)*((3/5)**0.5)
r2=1.6394*Z**(1/3)*((3/5)**0.5)
r3=1.2827*(1-0.2700*(N-Z)/A)*A**(1/3)*((3/5)**0.5)
r4=1.2331*(1-0.1461*(N-Z)/A+2.3301*1/A)*A**(1/3)*((3/5)**0.5)
# NN=A-A/(1.98+0.155*A**(2/3))
r5=1.6312*(1+0.0627*(N-(A-A/(1.98+0.155*A**(2/3))))/Z)*Z**(1/3)*((3/5)**0.5)

#%%
'E21  处理'
for i in range(len(data[:,0])):
    for j in range(len(E2)):
        if data[i, 0] == E2[j, 0] and data[i, 1] == E2[j, 1]:
            E21[i] = E2[j, 3]


'E41  处理'
for i in range(len(data[:,0])):
    for j in range(len(E4)):
        if data[i, 0] == E4[j, 0] and data[i, 1] == E4[j, 1]:
            E41[i] = E4[j, 3]

'E02  处理'
for i in range(len(data[:,0])):
    for j in range(len(E002)):
        if data[i, 0] == E002[j, 0] and data[i, 1] == E002[j, 1]:
            E02[i] = E002[j, 3]




#%%
Magic=np.array([2,8,20,28,50,82,126])
NT=np.min(np.abs(N.reshape(-1,1)-Magic),axis=1)
PT=np.min(np.abs(Z.reshape(-1,1)-Magic),axis=1)
#%%
ZDS=np.array([34,56,88])
ZD=np.min(np.abs(N.reshape(-1,1)-ZDS),axis=1)
NDS=np.array([34,56,88,134])
ND=np.min(np.abs(N.reshape(-1,1)-NDS),axis=1)
#%%
ss=(-1)**Z/2+(-1)**N/2
P=NT*PT/(NT+PT)
II=(N-Z)**2/A**2
Zm=np.zeros(len(A))
Nm=np.zeros(len(A))
#%%
for i in range(len(A)):
    if 8<N[i]<20:
        Nm[i]=1
    if 20 <= N[i] < 28:
        Nm[i] = 2
    if 28 <= N[i] < 50:
        Nm[i] = 3
    if 50 <= N[i] <82 :
        Nm[i] = 4
    if 82 <= N[i] < 126:
        Nm[i] = 5
    if N[i] > 126:
        Nm[i] = 6
for i in range(len(A)):
    if 8<=Z[i]<20:
        Zm[i]=1
    if 20 <= Z[i] < 28:
        Zm[i] = 2
    if 28 <= Z[i] < 50:
        Zm[i] = 3
    if 50 <= Z[i] <82 :
        Zm[i] = 4
    if 82 <= Z[i] < 126:
        Zm[i] = 5
    if Z[i] > 126:
        Zm[i] = 6
Zm1=Zm
Nm1=Nm
#%%

def index(data,Z,col,col1):
    return [x for (x,m) in enumerate(np.column_stack((data[:,col],data[:,col1]))) if m[0] != Z and m[1]!=Z]

dtaa=np.column_stack((Z,N,A,Sp,Sn,S2p,S2n,Saer,beata1,beata2,B,Blqu,B-Blqu,P,NT,PT,ZD,ND,E21,E41,E02,Rexp,Exp_error))
print(dtaa.shape)

# xuan=np.column_stack((E21,E41,E02))
# suoyin=np.all(xuan != 0, axis=1)
# # suoyin=index(dtaa,0,-3,-4)
# shuju=np.column_stack((Z,N,A,Rexp,Exp_error))[suoyin]
# dtaa=dtaa[suoyin]



#%%
dtaa=pd.DataFrame(dtaa)
dtaa[13]=dtaa[13].fillna(0)
daa=np.array(dtaa)







#%%

"Z,N,P,E21,E41,Rexp,beata,Exp_error"

# np.savetxt(path1+"机器学习数据test.txt",test,fmt='%20.8f')
# np.savetxt(path1+"机器学习数据train.txt",train,fmt='%20.8f ')

np.savetxt(path1+"机器学习数据合并.txt",daa,fmt='%20.8f')




