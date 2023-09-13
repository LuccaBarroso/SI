import numpy as np
import matplotlib.pyplot as plt

# Load data from the CSV file
dataEmg = np.loadtxt('data/EMG.csv', delimiter=',')

proporcao = 0.8
rodadas = 100

# Gerar os labels
y = np.repeat(np.arange(1, 6), 1000)
# y vai repetir esse padrao 10 vezes (50000)
y = np.tile(y, 10)

# Separa X1 e X2
x1 = dataEmg[:, 0]
x2 = dataEmg[:, 1]

def separarAleatoriamente(x1, x2, y):
    indices = np.random.permutation(len(x1))
    
    newX1 = x1[indices]
    newX2 = x2[indices]
    newY = y[indices]
    
    dadosTeste = {
        'x1': newX1[int(len(x1)*proporcao):],
        'x2': newX2[int(len(x2)*proporcao):],
        'y': newY[int(len(y)*proporcao):]
    }
    dadosTreino = {
        'x1': newX1[:int(len(x1)*proporcao)],
        'x2': newX2[:int(len(x2)*proporcao)],
        'y': newY[:int(len(y)*proporcao)]
    }
    

def mostrarMQO():


# Plot the data
plt.scatter(x1, x2, c=y, edgecolors='black', cmap='rainbow')



# Show the plot
plt.show()
