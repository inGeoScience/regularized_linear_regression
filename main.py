import numpy
import pandas


# 计算Cost Function
def compute_cost_function(X, y, w, b, lambda_):
    # 样本量
    m = X.shape[0]
    # 特征数量
    n = X.shape[1]
    # 初始化平方误差项与正则化项
    squared_error = 0
    regularization_term = 0
    # 计算平方误差项
    for i in range(m):
        squared_error += (numpy.dot(w, X[i]) + b - y[i])**2
    squared_error /= 2 * m
    # 计算正则化项
    for j in range(n):
        regularization_term += w[j]**2
    regularization_term = regularization_term * lambda_ / (2 * m)
    return squared_error + regularization_term


# 梯度下降
def gradient_descent(X, y, w, b, alpha, iters, lambda_):
    # 初始化临时参数向量与截距项
    tmp_w = numpy.zeros(X.shape[1])
    tmp_b = 0
    # 获取样本量与特征数
    m, n = X.shape
    # 初始化偏导数
    dj_dw = numpy.zeros(n)
    dj_db = 0
    # 进入梯度下降
    for iter_ in range(iters):
        # 计算偏导数（梯度）
        for i in range(m):
            error = numpy.dot(w, X[i]) + b - y[i]
            for j in range(n):
                dj_dw[j] += numpy.dot(error, X[i, j])
            dj_db += error
        dj_dw /= m
        dj_db /= m
        for j in range(n):
            dj_dw[j] += (lambda_ / m) * w[j]
        # 执行一次梯度下降
        tmp_w = w - alpha * dj_dw
        tmp_b = b - alpha * dj_db
        # 同步更新
        w = tmp_w
        b = tmp_b
        # 计算此次代价
        total_cost = compute_cost_function(X=X_train, y=y_train, w=w, b=b, lambda_=lambda_)
    return w, b, total_cost


# 导入数据
data = pandas.read_csv('ex1data2.txt', header=None, names=['Size', 'Bedrooms', 'Price'])
# Mean Normalization
for col in ['Size', 'Bedrooms']:
    data.loc[:, col] = (data.loc[:, col] - data.loc[:, col].mean()) / (data.loc[:, col].max() - data.loc[:, col].min())
data.loc[:, 'Price'] = data.loc[:, 'Price'].apply(lambda x: x / 10000)
# 定义训练样本的自变量2D array X_train与目标变量1d array y_train
X_train = data.iloc[:, 0:data.shape[1] - 1]  # 获得的是DataFrame
X_train = numpy.array(X_train.values)  # 这样获得一个2D array，如果转成matrix后面X_train[i]就会是一个2D array，不方便操作。
# X_train = numpy.matrix(X_train.values)
# numpy.array(X_train[i].flatten())
y_train = numpy.array(data.iloc[:, data.shape[1] - 1:data.shape[1]].values)
# 初始化w和b，w是一个1d array
w = numpy.zeros(X_train.shape[1])
b = 0
# 输入λ、α与迭代次数
lambda_ = float(input('Lambda:'))
alpha = float(input('alpha:'))
iters_ = int(input('iters:'))
# 进行正则化线性回归梯度下降求解
w, b, tt_cost = gradient_descent(X=X_train, y=y_train, w=w, b=b, alpha=alpha, iters=iters_, lambda_=lambda_)
# 输出结果
print(f'w:{w}\nb:{b}\ntotal_cost:{tt_cost}')
