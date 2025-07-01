# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division
# import os
import argparse
import pandas as pd
import numpy as np
from six.moves import range
import tensorflow as tf
import zhusuan as zs
import random
from sklearn.model_selection import train_test_split
# from examples import conf
# from examples.utils import dataset
# from examples.gaussian_process.utils import gp_conditional, RBFKernel
#%%
parser = argparse.ArgumentParser()
parser.add_argument('-n_z', default=100, type=int)
parser.add_argument('-n_particles', default=20, type=int)
parser.add_argument('-n_particles_test', default=100, type=int)
parser.add_argument('-batch_size', default=300, type=int)
parser.add_argument('-n_epoch', default=5000, type=int)
parser.add_argument('-dtype', default='float32', type=str,
                    choices=['float32', 'float64'])
parser.add_argument('-dataset', default='boston_housing', type=str,
                    choices=['boston_housing', 'protein_data'])
parser.add_argument('-lr', default=1e-2, type=float)

#%%
hps = parser.parse_args()
#%%
def standardize(data_train, data_test):
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    data_train_standardized = (data_train - mean) / std
    data_test_standardized = (data_test - mean) / std
    mean, std = np.squeeze(mean, 0), np.squeeze(std, 0)
    return data_train_standardized, data_test_standardized, mean, std
#%%
def unstandardize(data_train,data_test_standardized, data_pred_standardized):
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    data_test = data_test_standardized * std + mean
    data_pred = data_pred_standardized * std + mean
    return data_test , data_pred

#%%
class RBFKernel:

    def __init__(self, n_covariates, name='rbf_kernel', dtype=tf.float32):
        k_raw_scale = tf.get_variable('k_log_scale_{}'.format(name),
                                      [n_covariates], dtype,
                                      initializer=tf.zeros_initializer())
        self.k_scale = tf.nn.softplus(k_raw_scale)

    def __call__(self, x, y):
        '''
        Return K(x, y), where x and y are possibly batched.
        :param x: shape [..., n_x, n_covariates]
        :param y: shape [..., n_y, n_covariates]
        :return: Tensor with shape [..., n_x, n_y]
        '''
        batch_shape = tf.shape(x)[:-2]
        rank = x.shape.ndims
        assert_ops = [
            tf.assert_greater_equal(
                rank, 2,
                message='RBFKernel: rank(x) should be static and >=2'),
            tf.assert_equal(
                rank, tf.rank(y),
                message='RBFKernel: x and y should have the same rank')]
        with tf.control_dependencies(assert_ops):
            x = tf.expand_dims(x, rank - 1)
            y = tf.expand_dims(y, rank - 2)
            k_scale = tf.reshape(self.k_scale, [1] * rank + [-1])
            ret = tf.exp(
                -tf.reduce_sum(tf.square(x - y) / k_scale, axis=-1) / 2)
        return ret

    def Kdiag(self, x):
        '''
        Optimized equivalent of diag_part(self(x, x))
        '''
        if x.shape.ndims == 2:
            return tf.ones([tf.shape(x)[0]], dtype=x.dtype)
        else:
            return tf.ones([tf.shape(x)[0], tf.shape(x)[1]], dtype=x.dtype)


def gp_conditional(z, fz, x, full_cov, kernel, Kzz_chol=None):
    '''
    GP gp_conditional f(x) | f(z)==fz
    :param z: shape [n_z, n_covariates]
    :param fz: shape [n_particles, n_z]
    :param x: shape [n_x, n_covariates]
    :return: a distribution with shape [n_particles, n_x]
    '''
    n_z = int(z.shape[0])
    n_particles = tf.shape(fz)[0]

    if Kzz_chol is None:
        Kzz_chol = tf.cholesky(kernel(z, z))

    # Mean[fx|fz] = Kxz @ inv(Kzz) @ fz; Cov[fx|z] = Kxx - Kxz @ inv(Kzz) @ Kzx
    # With ill-conditioned Kzz, the inverse is often asymmetric, which
    # breaks further cholesky decomposition. We compute a symmetric one.
    Kzz_chol_inv = tf.matrix_triangular_solve(Kzz_chol, tf.eye(n_z))
    Kzz_inv = tf.matmul(tf.transpose(Kzz_chol_inv), Kzz_chol_inv)
    Kxz = kernel(x, z)  # [n_x, n_z]
    Kxziz = tf.matmul(Kxz, Kzz_inv)
    mean_fx_given_fz = tf.matmul(fz, tf.matrix_transpose(Kxziz))

    if full_cov:
        cov_fx_given_fz = kernel(x, x) - tf.matmul(Kxziz, tf.transpose(Kxz))
        cov_fx_given_fz = tf.tile(
            tf.expand_dims(tf.cholesky(cov_fx_given_fz), 0),
            [n_particles, 1, 1])
        fx_given_fz = zs.distributions.MultivariateNormalCholesky(
            mean_fx_given_fz, cov_fx_given_fz)
    else:
        # diag(AA^T) = sum(A**2, axis=-1)
        var = kernel.Kdiag(x) - \
            tf.reduce_sum(tf.matmul(
                Kxz, tf.matrix_transpose(Kzz_chol_inv)) ** 2, axis=-1)
        std = tf.sqrt(var)
        fx_given_fz = zs.distributions.Normal(
            mean=mean_fx_given_fz, std=std, group_ndims=1)
    return fx_given_fz
#%%
@zs.meta_bayesian_net(scope='model', reuse_variables=True)
def build_model(hps, kernel, z_pos, x, n_particles, full_cov=False):
    """
    Build the SVGP model.
    Note that for inference, we only need the diagonal part of Cov[Y], as
    ELBO equals sum over individual observations.
    For visualization etc we may want a full covariance. Thus the argument
    `full_cov`.
    """
    bn = zs.BayesianNet()
    Kzz_chol = tf.cholesky(kernel(z_pos, z_pos))
    fz = bn.multivariate_normal_cholesky(
        'fz', tf.zeros([hps.n_z], dtype=hps.dtype), Kzz_chol,
        n_samples=n_particles)
    # f(X)|f(Z) follows GP(0, K) gp_conditional
    fx_given_fz = bn.stochastic(
        'fx', gp_conditional(z_pos, fz, x, full_cov, kernel, Kzz_chol))
    # Y|f(X) ~ N(f(X), noise_level * I)
    noise_level = tf.get_variable(
        'noise_level', shape=[], dtype=hps.dtype,
        initializer=tf.constant_initializer(0.1))
    noise_level = tf.nn.softplus(noise_level)
    bn.normal('y', mean=fx_given_fz, std=noise_level, group_ndims=1)
    return bn


def build_variational(hps, kernel, z_pos, x, n_particles):
    bn = zs.BayesianNet()
    z_mean = tf.get_variable(
        'z/mean', [hps.n_z], hps.dtype, tf.zeros_initializer())
    z_cov_raw = tf.get_variable(
        'z/cov_raw', initializer=tf.eye(hps.n_z, dtype=hps.dtype))
    z_cov_tril = tf.matrix_set_diag(
        tf.matrix_band_part(z_cov_raw, -1, 0),
        tf.nn.softplus(tf.matrix_diag_part(z_cov_raw)))
    fz = bn.multivariate_normal_cholesky(
        'fz', z_mean, z_cov_tril, n_samples=n_particles)
    bn.stochastic('fx', gp_conditional(z_pos, fz, x, False, kernel))
    return bn


#%%
# Load data

path = r'H:\科研1\E02E03w\激发态2+\机器学习数据/'
data1 = pd.read_csv(path + '机器学习数据合并.txt', sep='\s+', header=None).to_numpy()
data2 = pd.read_csv(path + '机器学习数据外推.txt', sep='\s+', header=None).to_numpy()

# def index(data,Z,K,col,col1):
#     return [x for (x,m) in enumerate(np.column_stack((data[:,col],data[:,col1]))) if m[0] >= Z and m[1]>=K]
data1=pd.DataFrame(data1)
data1[20]=data1[20].fillna(0)
data2=pd.DataFrame(data2)
data2[20]=data2[20].fillna(0)


data1.columns = ["Z","N",'A','Sp','Sn','S2p','S2n',"Saer","beata1","beata2","B","Blqu","B-Blqu","P","NT","PT","ZD","ND","E21","E41","E02","exp",'err']
data2.columns = ["Z","N",'A','Sp','Sn','S2p','S2n',"Saer","beata1","beata2","B","Blqu","B-Blqu","P","NT","PT","ZD","ND","E21","E41","E02","exp",'err']

feats=["Z","N",'A','Sp','Sn','S2p','S2n',"Saer","beata1","beata2","B","Blqu","B-Blqu","P","NT","PT"]

x = np.array(data1[feats]).astype("float32")
X_test0= np.array(data1[feats]).astype("float32")
y = np.array(np.log10(data1["exp"])).astype("float32")
Y_test0 = np.array(np.log10(data1["exp"])).astype("float32")
# mode = np.array(data2["R_np"]).astype("float32")
X_train0, _, Y_train0, _= train_test_split(x, y, test_size=0.2, random_state=100)

X_waitui= np.array(data2[feats]).astype("float32")
Y_waitui = np.array(np.log10(data2["exp"])).astype("float32")



#%%
n_train, n_covariates = X_train0.shape
hps.dtype = getattr(tf, hps.dtype)

# Standardize data
x_train, x_test, _, _ = standardize(X_train0, X_test0)
y_train, y_test, mean_y_train, std_y_train = standardize(Y_train0, Y_test0)
#%%
# Build model
kernel = RBFKernel(n_covariates)
x_ph = tf.placeholder(hps.dtype, [None, n_covariates], 'x')
y_ph = tf.placeholder(hps.dtype, [None], 'y')
z_pos = tf.get_variable('z/pos', [hps.n_z, n_covariates], hps.dtype,initializer=tf.random_uniform_initializer(-1, 1))
n_particles_ph = n_particles_ph = tf.placeholder(tf.int32, [], 'n_particles')
batch_size = tf.cast(tf.shape(x_ph)[0], hps.dtype)
#%%
model = build_model(hps, kernel, z_pos, x_ph, n_particles_ph)
variational = build_variational(hps, kernel, z_pos, x_ph, n_particles_ph)
#%%
# ELBO = E_q log (p(y|fx)p(fx|fz)p(fz) / p(fx|fz)q(fz))
# So we remove p(fx|fz) in both log_joint and latent
def log_joint(bn):
        prior, log_py_given_fx = bn.cond_log_prob(['fz', 'y'])
        return prior + log_py_given_fx / batch_size * n_train

model.log_joint = log_joint

[var_fz, var_fx] = variational.query(
        ['fz', 'fx'], outputs=True, local_log_prob=True)
var_fx = (var_fx[0], tf.zeros_like(var_fx[1]))
lower_bound = zs.variational.elbo(
        model,
        observed={'y': y_ph},
        latent={'fz': var_fz, 'fx': var_fx},
        axis=0)
cost = lower_bound.sgvb()
optimizer = tf.train.AdamOptimizer(learning_rate=hps.lr)
infer_op = optimizer.minimize(cost)
#%%
# Prediction ops
model = model.observe(fx=var_fx[0], y=y_ph)
log_likelihood = model.cond_log_prob('y')
std_y_train = tf.cast(std_y_train, hps.dtype)
log_likelihood = zs.log_mean_exp(log_likelihood, 0) / batch_size - \
        tf.log(std_y_train)
yzong=model['y'].distribution.mean
y_pred_mean = tf.reduce_mean(model['y'].distribution.mean, axis=0)
pred_mse = tf.reduce_mean((y_pred_mean - y_ph) ** 2) * std_y_train ** 2
#%%
def infer_step(sess, x_batch, y_batch):
        fd = {
            x_ph: x_batch,
            y_ph: y_batch,
            n_particles_ph: hps.n_particles
        }
        return sess.run([infer_op, lower_bound], fd)[1]

def predict_step(sess, x_batch, y_batch):
        fd = {
            x_ph: x_batch,
            y_ph: y_batch,
            n_particles_ph: hps.n_particles_test
        }
        return sess.run([log_likelihood, pred_mse, y_pred_mean , y_ph,yzong], fd)

iters = int(np.ceil(x_train.shape[0] / float(hps.batch_size)))
test_freq = 10


#%%

def permutation_importance(sess, X_test, y_test, original_mse):
    importance_scores = []
    for feature_idx in range(X_test.shape[1]):
        # 复制测试集并打乱特定特征
        X_perturbed = X_test.copy()
        np.random.shuffle(X_perturbed[:, feature_idx])

        # 计算扰动后的MSE
        perturbed_mses = []
        for t in range(0, X_perturbed.shape[0], hps.batch_size):
            _, mse, _, _, _ = predict_step(
                sess,
                X_perturbed[t:t + hps.batch_size],
                y_test[t:t + hps.batch_size]
            )
            perturbed_mses.append(mse)
        perturbed_mse = np.mean(perturbed_mses)

        # 重要性得分 = 扰动后MSE - 原始MSE
        importance = perturbed_mse - original_mse
        importance_scores.append(importance)

    return np.array(importance_scores)


#%%
loop=500



for k in range(loop):

    X_train0, X_sheng, Y_train0, Y_sheng = train_test_split(x, y, test_size=0.2, random_state=random.randint(1, 500))

    # Standardize data
    x_train, x_test, _, _ = standardize(X_train0, X_test0)
    _, x_val, _, _ = standardize(X_train0, X_sheng)
    _, x_wai, _, _ = standardize(X_train0, X_waitui)
    y_train, y_test, mean_y_train, std_y_train = standardize(Y_train0, Y_test0)
    _, y_val, _, _ = standardize(Y_train0, Y_sheng)
    _, y_wai, _, _ = standardize(Y_train0, Y_waitui)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver(max_to_keep=150)
        for epoch in range(1, hps.n_epoch + 1):
            lbs = []
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]
            for t in range(iters):
                lb = infer_step(
                    sess,
                    x_train[t * hps.batch_size: (t + 1) * hps.batch_size],
                    y_train[t * hps.batch_size: (t + 1) * hps.batch_size])
                lbs.append(lb)
            if 10 * epoch % test_freq == 0:
                print('Epoch {}: Lower bound = {}'.format(epoch, np.mean(lbs)))

            if epoch % test_freq == 0:

                test_lls_test = []
                test_mses_test = []
                Y_t = 0
                Y_p = 0
                Y_Z = 0
                for t in range(0, x_test.shape[0], hps.batch_size):
                    ll, mse, yp, yt, yZ = predict_step(
                        sess,
                        x_test[t: t + hps.batch_size],
                        y_test[t: t + hps.batch_size])
                    test_lls_test.append(ll)
                    test_mses_test.append(mse)
                    if t == 0:
                        Y_t = yt.reshape(1, -1)
                        Y_p = yp.reshape(1, -1)
                        Y_Z = yZ
                    else:
                        Y_t = np.column_stack((Y_t, yt.reshape(1, -1)))
                        Y_p = np.column_stack((Y_p, yp.reshape(1, -1)))
                        Y_Z = np.column_stack((Y_Z, yZ))
                Y_all_test, Y_pred_test = unstandardize(Y_train0, Y_Z, Y_p)
                Y_test, _ = unstandardize(Y_train0, Y_t, Y_p)
                Y_all_T_test = np.transpose(Y_all_test)
                Y_pred_T_test = np.transpose(Y_pred_test)






                test_lls_train = []
                test_mses_train = []
                Y_t = 0
                Y_p = 0
                Y_Z = 0
                for t in range(0, x_train.shape[0], hps.batch_size):
                    ll, mse, yp, yt, yZ = predict_step(
                        sess,
                        x_train[t: t + hps.batch_size],
                        y_train[t: t + hps.batch_size])
                    test_lls_train.append(ll)
                    test_mses_train.append(mse)
                    if t == 0:
                        Y_t = yt.reshape(1, -1)
                        Y_p = yp.reshape(1, -1)
                        Y_Z = yZ
                    else:
                        Y_t = np.column_stack((Y_t, yt.reshape(1, -1)))
                        Y_p = np.column_stack((Y_p, yp.reshape(1, -1)))
                        Y_Z = np.column_stack((Y_Z, yZ))
                Y_all_train, Y_pred_train = unstandardize(Y_train0, Y_Z, Y_p)
                Y_train, _ = unstandardize(Y_train0, Y_t, Y_p)
                Y_all_T_train = np.transpose(Y_all_train)
                Y_pred_T_train = np.transpose(Y_pred_train)


                test_lls_val = []
                test_mses_val = []
                Y_t = 0
                Y_p = 0
                Y_Z = 0
                for t in range(0, x_val.shape[0], hps.batch_size):
                    ll, mse, yp, yt, yZ = predict_step(
                        sess,
                        x_val[t: t + hps.batch_size],
                        y_val[t: t + hps.batch_size])
                    test_lls_val.append(ll)
                    test_mses_val.append(mse)
                    if t == 0:
                        Y_t = yt.reshape(1, -1)
                        Y_p = yp.reshape(1, -1)
                        Y_Z = yZ
                    else:
                        Y_t = np.column_stack((Y_t, yt.reshape(1, -1)))
                        Y_p = np.column_stack((Y_p, yp.reshape(1, -1)))
                        Y_Z = np.column_stack((Y_Z, yZ))
                Y_all_val, Y_pred_val = unstandardize(Y_train0, Y_Z, Y_p)
                Y_val, _ = unstandardize(Y_train0, Y_t, Y_p)
                Y_all_T_val = np.transpose(Y_all_val)
                Y_pred_T_val = np.transpose(Y_pred_val)



                test_lls_wai = []
                test_mses_wai = []
                Y_t = 0
                Y_p = 0
                Y_Z = 0
                for t in range(0, x_wai.shape[0], hps.batch_size):
                    ll, mse, yp, yt, yZ = predict_step(
                        sess,
                        x_wai[t: t + hps.batch_size],
                        y_wai[t: t + hps.batch_size])
                    test_lls_wai.append(ll)
                    test_mses_wai.append(mse)
                    if t == 0:
                        Y_t = yt.reshape(1, -1)
                        Y_p = yp.reshape(1, -1)
                        Y_Z = yZ
                    else:
                        Y_t = np.column_stack((Y_t, yt.reshape(1, -1)))
                        Y_p = np.column_stack((Y_p, yp.reshape(1, -1)))
                        Y_Z = np.column_stack((Y_Z, yZ))
                Y_all_wai, Y_pred_wai = unstandardize(Y_train0, Y_Z, Y_p)
                Y_wai, _ = unstandardize(Y_train0, Y_t, Y_p)
                Y_all_T_wai = np.transpose(Y_all_wai)
                Y_pred_T_wai = np.transpose(Y_pred_wai)



                print('>> TEST')
                print('>> Test log likelihood = {}, rmse = {}'.format(
                    np.mean(test_lls_test), np.sqrt(np.mean(test_mses_test))))

                print('>> TRAIN')
                print('>> Train log likelihood = {}, rmse = {}'.format(
                    np.mean(test_lls_train), np.sqrt(np.mean(test_mses_train))))

                print('>> VAL')
                print('>> Val log likelihood = {}, rmse = {}'.format(
                    np.mean(test_lls_val), np.sqrt(np.mean(test_mses_val))))





                # if epoch >= 5500 :
                #     saver.save(sess, r'H:\科研1\核半径(BNN)\高斯bnn模型保存\gpr修改原版_Rnp_10F\my_model_{}'.format(epoch))
                #     np.savetxt(r'H:\科研1\核半径(BNN)\高斯bnn数据保存\gpr修改原版_Rnp_10F\Y_pred_model_{}_data_{:3.4f}.txt'.format(epoch,np.sqrt(np.mean(test_mses))),np.column_stack((Y_all_T,Y_pred_T,np.std(Y_all_T,axis=1))) ,fmt='%20.8f ')
                #保存当前模型

    print('>> NUM = {}'.format(k))
    if k==0:
        preed_test=Y_pred_T_test
        rmsee_test=np.sqrt(np.mean(test_mses_test))

        preed_train=Y_pred_T_train
        rmsee_train=np.sqrt(np.mean(test_mses_train))

        preed_val=Y_pred_T_val
        rmsee_val=np.sqrt(np.mean(test_mses_val))

        preed_wai=Y_pred_T_wai
        rmsee_wai=np.sqrt(np.mean(test_mses_wai))
    else:
        preed_test=np.column_stack((preed_test,Y_pred_T_test))
        rmsee_test = np.column_stack((rmsee_test,np.sqrt(np.mean(test_mses_test))))

        preed_train=np.column_stack((preed_train,Y_pred_T_train))
        rmsee_train = np.column_stack((rmsee_train,np.sqrt(np.mean(test_mses_train))))

        preed_val=np.column_stack((preed_val,Y_pred_T_val))
        rmsee_val = np.column_stack((rmsee_val,np.sqrt(np.mean(test_mses_val))))

        preed_wai=np.column_stack((preed_wai,Y_pred_T_wai))
        rmsee_wai = np.column_stack((rmsee_wai,np.sqrt(np.mean(test_mses_wai))))
#%%
pathh1 = r"H:\科研1\E02E03w\激发态2+\svgp\rms随比例变化\合并数据/"
pathh2 = r"H:\科研1\E02E03w\激发态2+\svgp\rms随比例变化\训练数据/"
pathh3 = r"H:\科研1\E02E03w\激发态2+\svgp\rms随比例变化\剩余数据/"
pathh4 = r"H:\科研1\E02E03w\激发态2+\svgp\rms随比例变化\外推数据/"



np.savetxt(pathh1 + 'pre_80_F' + "16" + '.txt',
           np.column_stack((preed_test, np.mean(preed_test, axis=1), np.std(preed_test, axis=1))), fmt='%10.8f  ')
np.savetxt(pathh1 + 'rms_80_F' + "16" + '.txt',
           np.column_stack((rmsee_test, np.mean(rmsee_test, axis=1), np.std(rmsee_test, axis=1))), fmt='%10.8f  ')

np.savetxt(pathh2 + 'pre_80_F' + "16" + '.txt',
           np.column_stack((preed_train, np.mean(preed_train, axis=1), np.std(preed_train, axis=1))), fmt='%10.8f  ')
np.savetxt(pathh2 + 'rms_80_F' + "16" + '.txt',
           np.column_stack((rmsee_train, np.mean(rmsee_train, axis=1), np.std(rmsee_train, axis=1))), fmt='%10.8f  ')

np.savetxt(pathh3 + 'pre_80_F' + "16" + '.txt',
           np.column_stack((preed_val, np.mean(preed_val, axis=1), np.std(preed_val, axis=1))), fmt='%10.8f  ')
np.savetxt(pathh3 + 'rms_80_F' + "16" + '.txt',
           np.column_stack((rmsee_val, np.mean(rmsee_val, axis=1), np.std(rmsee_val, axis=1))), fmt='%10.8f  ')

np.savetxt(pathh4 + 'pre_80_F' + "16" + '.txt',
           np.column_stack((preed_wai, np.mean(preed_wai, axis=1), np.std(preed_wai, axis=1))), fmt='%10.8f  ')
np.savetxt(pathh4 + 'rms_80_F' + "16" + '.txt',
           np.column_stack((rmsee_wai, np.mean(rmsee_wai, axis=1), np.std(rmsee_wai, axis=1))), fmt='%10.8f  ')


#%%

