import numpy as np

from matplotlib import pyplot as plt


#sample train data
x = np.array([[-2,4,-1],[4,1,-1],[1,6,-1],[2,4,-1],[6,2,-1]])
#sample labels
y = np.array([-1,-1,1,1,1])

def svm_sgd(X,Y):
    #SVM with Stochastic Gradient Descent

    #Weight matrix initialized with 0s
    w = np.zeros(len(X[0]))
    #learning rate
    eta = 1
    #number of iterations for learning
    epochs = 100000

    for epoch in range(1,epochs):
        for i, x in enumerate(X):
            #data incorrectly classified
            if(Y[i]*np.dot(X[i], w)) < 1:
                #update by learning rate and regularizer
                w = w + eta * (X[i] * Y[i]) + (-2 * (1/epoch)* w)
            #data correctly classified
            else:
                #update only by regularizer
                w = w + eta * (-2 * (1/epoch) * w)

    return w

w = svm_sgd(x,y)
print ("The wieght matrix formed ::")
print(w)

#sample test set
x_test = np.array([[2,2,-1],[4,3,-1]])
y_test = np.array([-1,1])

for d, sample in enumerate(x_test):
    r = np.dot(sample,w)
    if (r<0):
        print ( sample, "classified as -1")
        m = '_'
        c = 'red'
    else:
        print ( sample, "classified as +1")
        m = '+'
        c = 'green'
    plt.scatter(sample[0],sample[1], s=120, marker=m,color=c)


#plotting the train set for graphic display 
for d, sample in enumerate(x):
    if d<2:
        #plotting the -ve class
        plt.scatter(sample[0],sample[1], s=120, marker='_',linewidths=2, color='blue')
    else:
        #plotting the +ve class
        plt.scatter(sample[0],sample[1], s=120, marker='+', linewidth=2, color='blue')

 

# Print the hyperplane calculated by svm_sgd()
x2=[w[0],w[1],-w[1],w[0]]
x3=[w[0],w[1],w[1],-w[0]]

x2x3 =np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X,Y,U,V,scale=1, color='blue')
plt.show()

