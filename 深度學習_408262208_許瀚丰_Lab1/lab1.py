from pla_algorithm import pla, sign
import matplotlib.pyplot as plt
import numpy as np

def draw(dataset, test, weights, bias):#畫圖的method input(訓練資料, 測試資料, 權重(w1, w2), 偏差值)
    #---資料----
    trainning_data = [v[0] for v in dataset]    #所有訓練資料(x1, x2)
    trainning_data_ans = [v[1] for v in dataset]    #所有訓練資料的class(1 or -1)
    #---畫出訓練與測試資料，並畫上分類線性方程式---
    #---基本設定---
    plt.title("Training and Testing Data")   #表格標題
    plt.xlabel('x1')    #x軸標籤
    plt.ylabel('x2')    #y軸標籤

    #---畫圖
    for i in range(len(trainning_data)):    #遍歷所有訓練資料
        if trainning_data_ans[i] == 1:   #訓練資料的y=1的情況，為藍色圓圈
            trainning_pos = plt.scatter(trainning_data[i][0], trainning_data[i][1], c = 'black', marker = 'o')
        else:   #訓練資料的y=-1的情況，為紅色叉叉
            trainning_neg = plt.scatter(trainning_data[i][0], trainning_data[i][1], c = 'red', marker = 'x')

    for i in range(len(test)):      #遍歷所有測試資料
        if sign(weights, test[i], bias) == 1:   #測試資料的y=1的情況，為黑色三角形
            test_pos = plt.scatter(test[i][0], test[i][1], c = 'black', marker = '^')
        else:   #測試資料的y=-1的情況，為紅色三角形
            test_neg = plt.scatter(test[i][0], test[i][1], c = 'red', marker = '^')

    plt.legend([trainning_pos, trainning_neg, test_pos, test_neg], ["+1(Training)", "-1(Training)", "+1(Testing)", "-1(Testing)"], loc = "lower right")     #顯示圖示

    #---畫出分類線性方程式---
    l = np.linspace(-8, 7)      #設定線畫在x軸的-8~7之間
    plt.plot(l, -((weights[0] * l + bias)/ weights[1]), 'b-')   #將方程式w0*x0+w1*x1+b=0變為x1=-[(w0*x0+b)/w1]，並用藍色畫出來

    plt.show()      #畫出所有圖

if __name__ == '__main__':
    #---資料---
    #trainning data
    dataset = np.array([
    ((1, 0), 1),
    ((1, 3), -1),
    ((2, -6), 1),
    ((-1, -3), 1),
    ((-5, 5), -1),
    ((5, 2), 1),
    ((-2, 2), -1),
    ((-7, 2), -1),
    ((4, -4), 1),
    ((-5, -1), -1)
    ])
    #testing data
    test = np.array([
    (2, -4),
    (-5, 2),
    (-2, -2)
    ])
    weights, bias = pla(dataset)     #將訓練資料丟進PLA_algorithm訓練，回傳權重weights(w0, w1) 與偏差值bias
    print(f'w0 = {weights[0]} w1 = {weights[1]} bias = {bias}')     #輸出我們的weights與bias
    draw(dataset, test, weights, bias)      #呼叫畫圖的method
