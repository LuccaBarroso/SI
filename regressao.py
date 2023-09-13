import numpy as np
import matplotlib.pyplot as plt
import statistics

# ler dados
dataAero = np.loadtxt('data/aerogerador.dat', usecols=range(2))

proporcao = 0.8

# separar dados em x e y
x = dataAero[:,0]
y = dataAero[:,1]

regularizacao = 0.4

# plotar dados
plt.scatter(x,y,color='aqua', edgecolors='black', label='Dados')

def separarAleatoriamente(x, y):
    indices = np.random.permutation(len(x))
    tempX = x[indices]
    tempY = y[indices]

    xTreinamento = tempX[:int(len(x)*proporcao)]
    yTreinamento = tempY[:int(len(y)*proporcao)]
    xTeste = tempX[int(len(x)*proporcao) :]
    yTeste = tempY[int(len(y)*proporcao) :]

    return [xTreinamento, xTeste, yTreinamento, yTeste]

def mostrarMediaValoresObservaveis(xTreinamento, yTreinamento):
    yMedia = np.mean(yTreinamento)
    plt.plot(xTreinamento, np.ones(len(xTreinamento))*yMedia, color='red', label='Média')


def EQMMediaValoresObservaveis(xTreinamento, yTreinamento, xTeste, yTeste):
    yMedia = np.mean(yTreinamento)

    # EQM
    return np.mean(abs(yTeste - yMedia) ** 2)
    
def mostrarMQO(xTreinamento, yTreinamento):
    # Criar matriz X
    X = np.concatenate((np.ones((len(xTreinamento),1)),xTreinamento.reshape(-1,1)),axis=1)
    
    # Estimação do modelo
    B = np.linalg.pinv(X.T@X)@X.T@yTreinamento

    # Criar eixo x
    tamanho = int(np.max(xTreinamento))
    x_axis = np.linspace (0,tamanho,tamanho)
    x_axis.shape = (len(x_axis),1)
    
    # Criar matriz X_new
    ones = np.ones((len(x_axis),1))
    X_new = np.concatenate((ones,x_axis), axis=1)

    # Criar matriz Y_pred
    Y_pred = X_new@B

    # Plotar reta
    plt.plot(x_axis,Y_pred, color='green', label='MQO')

def EQMMQO(xTreinamento, yTreinamento, xTeste, yTeste):
    # Criar matriz X
    X = np.concatenate((np.ones((len(xTreinamento),1)),xTreinamento.reshape(-1,1)),axis=1)
    
    # Estimação do modelo
    B = np.linalg.pinv(X.T@X)@X.T@yTreinamento

    # Criar matriz X_new
    ones = np.ones((len(xTeste),1))
    X_new = np.concatenate((ones,xTeste.reshape(-1,1)), axis=1)

    # Criar matriz Y_pred
    Y_pred = X_new@B

    # EQM
    return np.mean(abs(yTeste - Y_pred) ** 2)
  

def mostrarMQORegularizado(xTreinamento, yTreinamento):
    # Criar matriz X
    X = np.concatenate((np.ones((len(xTreinamento),1)),xTreinamento.reshape(-1,1)),axis=1)
    
    # Estimação do modelo por tikonov
    B = np.linalg.pinv(X.T@X + regularizacao*np.identity(2))@X.T@yTreinamento


    # Criar eixo x
    tamanho = int(np.max(xTreinamento))
    x_axis = np.linspace (0,tamanho,tamanho)
    x_axis.shape = (len(x_axis),1)

    # Criar matriz X_new
    ones = np.ones((len(x_axis),1))
    X_new = np.concatenate((ones,x_axis), axis=1)

    # Criar matriz Y_pred
    Y_pred = X_new@B
    
    # Plotar reta
    plt.plot(x_axis,Y_pred, color='purple', label='MQO Regularizado')

def EQMMQORegularizado(xTreinamento, yTreinamento, xTeste, yTeste, reg = 0.1):
    # Criar matriz X
    X = np.concatenate((np.ones((len(xTreinamento),1)),xTreinamento.reshape(-1,1)),axis=1)

    # Estimação do modelo por tikonov
    B = np.linalg.pinv(X.T@X + reg*np.identity(2))@X.T@yTreinamento

    # Criar matriz X_new
    ones = np.ones((len(xTeste),1))
    X_new = np.concatenate((ones,xTeste.reshape(-1,1)), axis=1)

    # Criar matriz Y_pred
    Y_pred = X_new@B

    # EQM
    return np.mean(abs(yTeste - Y_pred) ** 2)

def executarTeste(qnt):
    eqmsMVO = []
    eqmsMQO = []
    for i in range(qnt):
      [xTreinamento, xTeste, yTreinamento, yTeste] = separarAleatoriamente(x, y)

      eqmsMVO.append(EQMMediaValoresObservaveis(xTreinamento, yTreinamento, xTeste, yTeste))
      eqmsMQO.append(EQMMQO(xTreinamento, yTreinamento, xTeste, yTeste))
      eqmsMQO.append(EQMMQORegularizado(xTreinamento, yTreinamento, xTeste, yTeste))

    mostrarValores(eqmsMVO, "MVO")
    mostrarValores(eqmsMQO, "MQO")
    mostrarValores(eqmsMQO, "MQO Regularizado")
    
def encontrarMelhorRegularizacao(qntTestes):
    result = []
    for i in range(qntTestes):
        [xTreinamento, xTeste, yTreinamento, yTeste] = separarAleatoriamente(x, y)
        menor = 10
        menorEqm = 10000000
        for i in np.arange(0.1, 1, 0.1):
            cur = (EQMMQORegularizado(xTreinamento, yTreinamento, xTeste, yTeste, i))
            if(cur < menorEqm):
                menorEqm = cur
                menor = i
        result.append(menor)
    print("Melhor regularização: " + str(statistics.mode(result)))
        
        
        
encontrarMelhorRegularizacao(1000)
        
    
def mostrarValores(eqms, nome):
    print("-=-=-=-=-=-" + nome + "-=-=-=-=-=-")
    print("EQM: " + str(np.mean(eqms)))
    print("EQM Máximo: " + str(np.max(eqms)))
    print("EQM Mínimo: " + str(np.min(eqms)))
    print("EQM Desvio Padrão: " + str(np.std(eqms)))


executarTeste(1000)


[xTreinamento, xTeste, yTreinamento, yTeste] = separarAleatoriamente(x, y)
mostrarMediaValoresObservaveis(xTreinamento, yTreinamento)
mostrarMQO(xTreinamento, yTreinamento)
mostrarMQORegularizado(xTreinamento, yTreinamento)
        




# Mostrar gráfico
plt.show()