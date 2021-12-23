from pla_algorithm import check_all, pla, sign, read_data
import matplotlib.pyplot as plt
import numpy as np

def draw(dataset, test, weights, bias): #畫圖的method input(訓練資料, 測試資料, 權重(w1, w2), 偏差值)
    #---資料----
    trainning_data = [v[0] for v in dataset]    #所有訓練資料(x1, x2)
    trainning_data_ans = [v[1] for v in dataset]    #訓練資料的class(1 or -1)
    test_data = [v[0] for v in test]    #所有測試資料(x1, x2)
    test_data_ans = [v[1] for v in test]    #測試資料的class(1 or -1)

    #---figure1設定---
    plt.figure(figsize = (15, 5))   #設定圖視窗大小
    plt.figure(1)   #第一張圖(figure1)

    #---畫出訓練資料與測試資料---
    #---基本設定---
    plt.subplot(1, 2, 1)    #figure1的左邊(1行，2列，第一張圖)
    plt.title("Iris")   #表格標題
    plt.xlabel('sepal length')  #x軸標籤
    plt.ylabel('sepal width')   #y軸標籤

    #---畫圖---
    for i in range(len(trainning_data)):    #遍歷訓練資料
        if trainning_data_ans[i] == 1:  #訓練資料的y=1的情況，為藍色圓圈
            setosa = plt.scatter(trainning_data[i][0], trainning_data[i][1], c = 'blue', marker = 'o')
        else:   #訓練資料的y=-1的情況，為橘色叉叉
            non_setosa = plt.scatter(trainning_data[i][0], trainning_data[i][1], c = 'orange', marker = 'x')

    for i in range(len(test_data)):     #遍歷所有測試資料
        test = plt.scatter(test_data[i][0], test_data[i][1], c = 'green', marker = '^')     #測試資料皆為綠色三角形

    plt.legend([setosa, non_setosa, test], ["Setosa", "Non_setosa", "Test"], loc = "upper right")   #顯示圖示

    #---畫出測試資料---
    #---基本設定---
    plt.subplot(1, 2, 2)    #figure1的左邊(1行，2列，第二張圖)
    plt.title("Iris")   #表格標題
    plt.xlabel('sepal length')  #x軸標籤
    plt.ylabel('sepal width')   #y軸標籤

    #---畫圖---
    for i in range(len(test_data)):     #遍歷測試資料
        if test_data_ans[i] == 1:   #測試資料的y=1的情況，為藍色圓圈 
            setosa = plt.scatter(test_data[i][0], test_data[i][1], c = 'blue', marker = 'o')
        else:   #測試資料的y=-1的情況，為橘色叉叉
            non_setosa = plt.scatter(test_data[i][0], test_data[i][1], c = 'orange', marker = 'x')

    plt.legend([setosa, non_setosa], ["Setosa", "Non_setosa"], loc = "upper right")     #顯示圖示
    
    #---figure2設定---
    plt.figure(figsize = (15, 5))   #設定視窗大小
    plt.figure(2)   #第二張圖(figure2)

    #---畫出訓練與測試資料，並畫上分類線性方程式---
    #---基本設定---
    plt.subplot(1, 2, 1)    #figure1的左邊(1行，2列，第一張圖)
    plt.title("Training and Testing Data")   #表格標題
    plt.xlabel('sepal length')  #x軸標籤
    plt.ylabel('sepal width')   #y軸標籤

    #---畫圖---
    for i in range(len(trainning_data)):    #遍歷所有訓練資料
        if trainning_data_ans[i] == 1:  #訓練資料的y=1的情況，為藍色圓圈
            setosa = plt.scatter(trainning_data[i][0], trainning_data[i][1], c = 'blue', marker = 'o')
        else:   #訓練資料的y=-1的情況，為橘色叉叉
            non_setosa = plt.scatter(trainning_data[i][0], trainning_data[i][1], c = 'orange', marker = 'x')

    for i in range(len(test_data)):     #遍歷所有測試資料
        if test_data_ans[i] == 1:   #測試資料的y=1的情況，為綠色三角形
            test_setosa = plt.scatter(test_data[i][0], test_data[i][1], c = 'green', marker = '^')
        else:   #測試資料的y=-1的情況，為洋紅色星星
            test_non_setosa = plt.scatter(test_data[i][0], test_data[i][1], c = 'magenta', marker = '*')
    plt.legend([setosa, non_setosa, test_setosa, test_non_setosa], ["Setosa", "Non_setosa", "Test_setosa", "Test_non_setosa"], loc = "upper right")     #顯示圖示

    #---畫出分類線性方程式---
    l = np.linspace(4, 8)   #設定線畫在x軸的4~8之間
    plt.plot(l, -((weights[0] * l + bias)/ weights[1]), 'b-')   #將方程式w0*x0+w1*x1+b=0變為x1=-[(w0*x0+b)/w1]，並用藍色畫出來

    #---畫出測試準確率圓餅圖---
    #---基本設定---
    plt.subplot(1, 2, 2)    #figure1的左邊(1行，2列，第二張圖)
    plt.title('Predictive Accuracy')    #圓餅圖標題
    label = "Right", "Wrong"    #圓餅圖標籤
    ans = [0, 0]    #預測正確與錯誤次數，ans[0]為正確次數，ans[1]為錯誤次數
    for i in range(len(test_data)):     #遍歷所有測試資料
        if sign(weights, test_data[i], bias) == test_data_ans[i]:   #計算測試資料帶入分類線性方程式的sign值，並與其的分類做比較
            ans[0] += 1     #預測成功，正確次數加一
        else:
            ans[1] += 1     #預測失敗，錯誤次數加一
    plt.pie(ans, labels = label, autopct = "%1.1f%%")   #設定圓餅圖資料(正確與錯誤次數, 名稱, 圓餅圖精準度設定)

    plt.show()      #畫出所有圖
    return ans[0] / sum(ans)    #回傳預測準確度

if __name__ == "__main__":
    dataset = read_data("Iris_trainning.txt")   #dataset存入Iris_trainning.txt中的資料
    test = read_data("Iris_test.txt")   #test存入Iris_test.txt中的資料
    weights, bias = pla(dataset)    #將訓練資料丟進PLA_algorithm訓練，回傳權重weights(w0, w1) 與偏差值bias
    print(f"w0 = {weights[0]} w1 = {weights[1]} b = {bias}")    #輸出我們的weights與bias
    accuracy = draw(dataset, test, weights, bias)   #呼叫畫圖的method並將預測準確度存入accuracy
    print(f"Accuracy = {accuracy * 100}%")  #輸出預測準確度
