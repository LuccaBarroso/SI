import numpy as np
import matplotlib.pyplot as plt

# Load data from the CSV file
Data = np.loadtxt('data/EMG.csv', delimiter=',')

N,p = Data.shape
k = 0

neutro = np.tile(np.array([[1,-1,-1,-1,-1]]),(1000,1)) 
sorrindo = np.tile(np.array([[-1,1,-1,-1,-1]]),(1000,1)) 
aberto = np.tile(np.array([[-1,-1,1,-1,-1]]),(1000,1)) 
surpreso = np.tile(np.array([[-1,-1,-1,1,-1]]),(1000,1)) 
rabugento = np.tile(np.array([[-1,-1,-1,-1,1]]),(1000,1)) 

Y = np.tile(np.concatenate((neutro,sorrindo,aberto,surpreso,rabugento)),(10,1))

x1 = Data[:, 0]
x2 = Data[:, 1]

X = np.concatenate((x1.reshape(-1,1),x2.reshape(-1,1)),axis=1)

# mostrar dados
# Esse Y é só para plotar com cmap
Yplot = np.repeat(np.arange(1, 6), 1000)
Yplot = np.tile(Yplot, 10)

plt.scatter( X[:,0], X[:,1], c=Yplot, edgecolors='black', cmap='rainbow')

X = np.concatenate((
    np.ones((N,1)),X
),axis=1)

X_treino = X[0:int(N*.8),:]
Y_treino = Y[0:int(N*.8),:]

X_teste = X[int(N*.8):,:]
Y_teste = Y[int(N*.8):,:]

lb = 0.1
W_hat = np.linalg.pinv(X_treino.T@X_treino)@X_treino.T@Y_treino
W_hat_r = np.linalg.pinv(X_treino.T@X_treino + lb*np.eye(3))@X_treino.T@Y_treino

Y_hat = X_teste@W_hat

discriminante = np.argmax(Y_hat,axis=1)
discriminante2 = np.argmax(Y_teste,axis=1)

acertos = discriminante==discriminante2
print(np.count_nonzero(acertos)/10000)


plt.xlim(-100,4100)
plt.ylim(-100,4100)
plt.show()
bp=1




# Show the plot
plt.show()
