# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math, time
import datetime
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

# fixar random seed para se puder reproduzir os resultados
from keras.optimizers import Nadam
from sklearn.preprocessing import MinMaxScaler

seed = 9
np.random.seed(seed)


# Carregar os dados do .csv
def get_ad_data(normalized=0, file_name=None):
    activity = pd.read_csv(file_name, header=0)
    df = pd.DataFrame(activity)  # Criar dataFrame
    # df.drop(df.columns[[0]], axis=1, inplace=True)  # Largar coluna com data
    if normalized == 1:
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = pd.DataFrame(scaler.fit_transform(df))
        return dataset
    return df


# Comments Prof:
# função load_data do lstm.py configurada para aceitar qualquer número de parametros
# o último atributo é que fica como label (resultado)
# stock é um dataframe do pandas (uma especie de dicionario + matriz)
# seq_len é o tamanho da janela a ser utilizada na serie temporal
#
# Comments grupo:
# Função alterada para o problema (separação esntre dados de treino e teste)
def load_data(df_dados, janela):
    qt_atributos = len(df_dados.columns)
    mat_dados = df_dados.values  # converter dataframe para matriz (lista com lista de cada registo)
    tam_sequencia = janela + 1
    res = []
    for i in range(len(mat_dados) - janela):  # numero de registos - tamanho da sequencia
        res.append(mat_dados[i: i + tam_sequencia])
    res = np.array(res)  # dá como resultado um np com uma lista de matrizes (janela deslizante ao longo da serie)
    qt_casos_treino = 24  # dois anos de treino, um de teste
    train = res[:qt_casos_treino, :]
    x_train = train[:, :-1]  # menos um registo pois o ultimo registo é o registo a seguir à janela
    y_train = train[:, -1][:, -1]  # para ir buscar o último atributo para a lista dos labels
    x_test = res[qt_casos_treino:, :-1]
    y_test = res[qt_casos_treino:, -1][:, -1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], qt_atributos))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], qt_atributos))
    return [x_train, y_train, x_test, y_test]


# imprime um grafico com os valores de teste e com as correspondentes tabela de previsões
def print_series_prediction(y_test, predic):
    diff = []
    racio = []
    for i in range(len(y_test)):  # para imprimir tabela de previsoes
        racio.append((y_test[i] / predic[i]) - 1)
        diff.append(abs(y_test[i] - predic[i]))
        print('valor: %f ---> Previsão: %f Diff: %f Racio: %f' % (y_test[i], predic[i], diff[i], racio[i]))
    plt.plot(y_test, color='blue', label='y_test')
    plt.plot(predic, color='red', label='prediction')  # este deu uma linha em branco
    plt.plot(diff, color='green', label='diff')
    plt.plot(racio, color='yellow', label='racio')
    plt.legend(loc='upper left')
    plt.show()


# util para visualizar a topologia da rede num ficheiro em pdf ou png
def print_model(model, fich):
    from keras.utils import plot_model
    plot_model(model, to_file=fich, show_shapes=True, show_layer_names=True)



def build_model3(janela):
    model = Sequential()
    model.add(LSTM(30, input_shape=(janela, 14), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(20, input_shape=(janela, 14), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(10, input_shape=(janela, 14), return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation="relu", kernel_initializer="normal"))
    model.add(Dense(1, activation="linear", kernel_initializer="normal"))
    model.compile(loss='mse', optimizer='nadam', metrics=['mse', 'accuracy'])
    return model



def load_ad_dataset():
    #nornalizado
    return get_ad_data(1, 'adData.csv')



def print_history_accuracy(history):
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def print_history_loss(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    df = load_ad_dataset()
    print("df", df.shape)
    janela = 2  # tamanho da Janela deslizante um ano
    X_train, y_train, X_test, y_test = load_data(df, janela)
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)
    model = build_model3(janela)
    history = model.fit(X_train, y_train, batch_size=1, validation_data=(X_test, y_test), epochs=200, verbose=2,
                        shuffle=False)
    print_history_loss(history)
    print_model(model, "lstm_model.png")
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    testScore = model.evaluate(X_test, y_test, verbose=0)
    print(model.metrics_names)
    p = model.predict(X_test)
    predic = np.squeeze(
        np.asarray(p))  # para transformar uma matriz de uma coluna e n linhas em um np array de n elementos
    print_series_prediction(y_test, predic)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
