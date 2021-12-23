import matplotlib.pyplot as plt
import numpy as np
#---讀檔method---
def read_data(input_data):          #傳入檔案位置
    f = open(input_data, 'r')       #讀取檔案
    dataset = []                    #命名dataset為一個list
    for line in f.readlines():      #一行一行讀檔案
        line = line.split(',')      #將檔案利用逗號分開並存成list
        line[0] = float(line[0])    #將line[0](x1) 的string 轉型為float
        line[1] = float(line[1])    #將line[1](x2) 的string 轉型為float
        line[2] = int(line[2])      #將line[2](y) 的string 轉型為int
        x = line[:2]                #將前兩個欄位(x1, x2)儲存到x中
        y = line[2]                 #將最後一格欄位(-1 or 1)存到y中
        data = []                   #建一個空的list存資料
        data.append(tuple(x))       #將x轉型成tuple並append到data中
        data.append(y)              #將y append到data中
        dataset.append(data)        #將data append到 dataset中
    f.close()                       #將剛剛開啟的檔案關閉
    dataset = np.array(dataset)     #將dataset轉型成np的array
    return dataset                  #回傳dataset

#---判斷正負---
def sign(weights, x, bias):             #傳入weights(w1, w2), x(x1, x2), bias
    if np.dot(weights, x) + bias > 0:   #weights與x的內積 + bias > 0，回傳1
        return 1
    else:                               #weights與x的內積 + bias <= 0，回傳-1
        return -1

#---判斷是否全部的data的x(x1, x2)內積w(w1, w2)的sign是否與data的y(-1 or 1)相同，並回傳不相同的數量，用來控制學習率---
def check_all(dataset, weights, bias):      #傳入dataset((w1, w2), y), weights(w1, w2), bias
    number_of_not_matching = 0              #計算內積結果不相同的數量
    for x, y in dataset:                    #x = (x1, x2), y = y
        if sign(weights, x, bias) != y:     #w(w1, w2)內積x(x1, x2) + b 的 sign不等於y，則number_of_not_matching + 1
            number_of_not_matching += 1
    return number_of_not_matching           #回傳number_of_not_matching的結果

#---pla algorithm---
def pla(dataset):                           #傳入dataset((w1, w2), y)
    epoch_count = 1                         #設定目前世代
    epoch_max = len(dataset)                #設定最大世代
    weights = np.random.randn(2)            #隨機拿兩個數字當作weights(w1, w2)
    bias = np.random.randn()                #隨機拿一個數字當作bias

    #---利用check_all這個method 判斷是否所有dataset的x(x1, x2)內積weights加上bias的sign是否全都正確，若正確會回傳0---
    #---將最大世代設為dataset的數量 + 5 ，以防非線性可分離的情況，導致無窮迴圈---
    while check_all(dataset, weights, bias) != 0 and epoch_count <= epoch_max:
        #將learning_rate(0~1)設為(錯誤的次數 / dataset數量) -> 錯誤數量 ∝ learning rate
        learning_rate = check_all(dataset, weights, bias) / len(dataset)

        for x, y in dataset:    #x = (x1, x2), y = y
            x = np.array(x)     # list to array
            if sign(weights, x, bias) != y:     #weights(w1, w2)內積x(x1, x2)+b的sign不等於y，調整weights和bias
                #new_weights(new_w1, new_w2) = old_weights(old_w1, old_w2) + learning_rate(0~1) * y(-1 or 1) * x(x1, x2)
                weights = weights + learning_rate * y * x

                bias = bias + learning_rate * y     #new_bias = old_bias + learning_rate(0~1) * y(-1 or 1)

        epoch_count += 1    #世代數 + 1

    return weights, bias    #回傳 weights(w1, w2) 與 bias
