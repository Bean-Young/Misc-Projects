import re
import numpy as np

def matrix_minus(matrix1,matrix2):
    """
    Subtract two 1*n matrices
    The result returns a 1 * n matrix
    :param matrix1:a 1*n matrix
    :param matrix2:a 1*n matrix

    :return: new_matrix: matrix1-matrix2
    """

    new_matrix =  [matrix1[i] - matrix2[i] for i in range(len(matrix1)) ]
    return new_matrix

def lassoUseCd(X, y, lambdas=0.1, max_iter=1000, tol=1e-4):
    """
    Lasso regression, using coordinate descent method
    :param X: Training Dataset (a ten dimensional vector)
    :param y: Target label value (1 or 2, binary classification)
    :param lambdas: Penalty term coefficient (Default to 0.1)
    :param max_iter: Maximum number of iterations (Default to 1000)
    :param tol: Tolerance value for variation (Default to 0.0001)

    :return: W: Weight coefficient
    """
    # Global variable declaration
    global theory_V
    global datas
    global itera

    # Initialize W as a zero vector
    W = np.zeros(X.shape[1])
    for it in range(max_iter):
        done = True
        # Traverse all independent variables
        for i in range(0, len(W)):
            # Record the previous round of coefficients
            w = W[i]
            # Find the optimal coefficient under current conditions
            W[i] = down(X, y, W, i, lambdas)
            # Continue cycling if one of the coefficient changes does not reach its tolerance value
            if (np.abs(w - W[i]) > tol):
                done = False
        # All coefficients do not change much, end the cycle
        norm=np.linalg.norm(np.abs(matrix_minus(W,theory_V)))
        if (done) and (norm>datas[-1]):
            break
        print(it+1,'\t',norm)
        itera.append(it+1)
        datas.append(norm)
    return W

def down(X, y, W, index, lambdas=0.1):
    """
    Find the optimal coefficient
    cost(W) = (x1 * w1 + x2 * w2 + ... - y)^2 / 2n + ... + λ(|w1| + |w2| + ...)
    Assuming w1 is a variable and all other values are constants, the cost function of w1 is a quadratic function of one variable, which can be written as follows:
    cost(w1) = (a * w1 + b)^2 / 2n + ... + λ|w1| + c (a, b, c, λ  All are constants)
    =>Unfolding
    cost(w1) = aa / 2n * w1^2 + (ab / n) * w1 + λ|w1| + c (aa, ab, c, λ All are constants)

    :param X: Training Dataset (a ten dimensional vector)
    :param y: Target label value (1 or 2, binary classification)
    :param W: Weight coefficient
    :param index: Index of w
    :param lambdas: Penalty term coefficient (Default to 0.1)

    :return: w: The optimal coefficient under current conditions
    """

    # The sum of coefficients of the expanded second-order term
    aa = 0
    # The sum of coefficients of the expanded first-order term
    ab = 0
    for i in range(X.shape[0]):
        # The coefficient of the first_order term in parentheses
        a = X[i][index]
        # Coefficients of constant terms in parentheses
        b = X[i][:].dot(W) - a * W[index] - y[i]
        # The coefficient of the expanded second-order term is the sum of the squares of the coefficients of the first-order term in parentheses
        aa = aa + a * a
        # The coefficient of the expanded first-order term is obtained by multiplying the coefficient of the first-order term in parentheses by the sum of the constant terms in parentheses
        ab = ab + a * b
    # As it is a univariate quadratic function, when the derivative is zero, the function value is the minimum value, and only the second-order coefficient, first-order coefficient, and constant λ
    return det(aa, ab, X.shape[0], lambdas)

def det(aa, ab, n, lambdas=0.1):
    """
    Calculate w through the derivative of the cost function, and when w=0, it is not differentiable
    det(w) = (aa / n) * w + ab / n + λ = 0 (w > 0)
    => w = - (ab / n + λ) / (aa / n)

    det(w) = (aa / n) * w + ab / n - λ = 0 (w < 0)
    => w = - (ab / n - λ) / (aa / n)

    det(w) = NaN (w = 0)
    => w = 0

    :param aa: Sum of coefficients of second-order terms
    :param ab: Sum of coefficients of the expanded first-order term
    :param n: Dimension of vectors
    :param lambdas: Penalty term coefficient (Default to 0.1)

    :return: w:The optimal coefficient under current conditions
    """
    w = - (ab / n + lambdas) / (aa / n)
    if w < 0:
        w = - (ab / n - lambdas) / (aa / n)
        if w > 0:
            w = 0
    return w

def read_file():
    """
    Read the dataset and process it as X(Training Dataset) and y (Target label value)

    """
    with open('./covtype.libsvm.binary.scale.txt', 'r', encoding='utf-8') as f1:
        for num, line in enumerate(f1.readlines()):
            Y=int(line[0:1:1])
            line=line[2::]
            line='{'+line[:-12:]+'}'
            line=re.sub(' ',',',line)
            #print(line) (test)
            dic_data=eval(line)
            #Determine if all ten dimensions have values
            if sum(dic_data.keys())==55:
                X.append(list(dic_data.values()))
                y.append(Y)

def getx(X,y):
    """
    Find x*,through external libraries scikit-learn
    :param X: Training Dataset
    :param y: Target label value
    """
    from sklearn.linear_model import Lasso
    # Initialize Lasso regression, default to using coordinate descent method
    lasso = Lasso(alpha=0.1,fit_intercept=False)
    # Fitting linear models
    lasso.fit(X, y)
    global  theory_V
    # Weight coefficient
    theory_V = lasso.coef_

def output(itera_s,data_s):
    """
    Output Convergence Graph
    :param row: Data of x
    :param column: Data of y
    """
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(len(data_s)):
        datas=data_s[i]
        itera=itera_s[i]
        l=datas[-1]
        datas=datas[0:len(datas)-1]
        plt.plot(itera,datas,label='lambdas='+str(l))
    plt.xlabel('Iterations')
    plt.ylabel('||xk-x*||')
    plt.legend()
    plt.savefig('Different lambdas for LASSO problem.png')

if __name__ == '__main__':

    #X: Training Dataset y: Target label value datas: ||xk-x*||
    X=[]
    y=[]
    datas=[]
    itera=[]
    data_s=[]
    itera_s=[]

    read_file()

    #Transforming X, y into matrices
    X = np.array(X)
    y = np.array(y)

    # Determine if the number of samples X and y is consistent（test）
    #if X.shape[0] != y.shape[0]:
    #    raise ValueError("Input variables have inconsistent numbers of samples")

    getx(X,y)

    for i in range(0,3):
        lambda_s=i/10
        print('lambda=',lambda_s)
        lassoUseCd(X,y,lambdas=lambda_s,max_iter=300)
        itera_s.append(itera)
        datas.append(lambda_s)
        data_s.append(datas)
        datas=[]
        itera=[]


    output(itera_s,data_s)
