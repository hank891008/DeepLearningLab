import pandas as pd
import numpy as np

#如果input是2就回傳1，其他(在此只會有5)時就回傳0
trans1 = lambda x: 1 if x == 2 else 0

#如果input是1就回傳2，其他(在此只會為0)時就回傳5
trans2 = lambda x: 2 if x == 1 else 5

#回傳input是否大於等於0.5
predict = lambda x: x >= 0.5

#改良版sigmoid function，解決overflow問題
def logistic(n):
    return np.array(0.5 * (1 + np.tanh(0.5 * n)))

#讀取train.csv中的資料並整理成我要形式
def read_training_data():
    data = pd.read_csv('train.csv')
    data = data.values.tolist()
    dataset = []
    for items in data:
        items = list(map(int, items))
        y = trans1(items[0])
        x = items[1:]
        tmp = []
        tmp.append(tuple(x))
        tmp.append(y)
        dataset.append(tmp)
    dataset = np.array(dataset, dtype = object)
    return dataset

#將data整理為Batch version所要求的形式
def adjustment(dataset):
    x = []
    y = []
    for i in dataset:
        x.append(i[0])
        y.append(i[1])
    x = np.transpose(np.array(x, dtype = np.float64))
    y = np.array([y])
    return x, y

#計算誤差量，此為average_of_the_cross_entropy_loss
def average_of_the_cross_entropy_loss(y_head, Y):
    #設定esp使得我的y_head只會在(0 + esp, 1 - esp)之間，防止log(0)會沒有定義的情況
    epsilon = 1e-9
    y_head = np.clip(y_head, 0 + epsilon, 1 - epsilon)
    
    #loss function的avg
    loss = -np.mean(Y * np.log(y_head) + (1 - Y) * np.log(1 - y_head))
    
    return loss

#此logistic_regression為batch的版本
def logistic_regression(learning_rate, max_epoch, tau, datas, test):
    #輸入train之前各項數據與全重
    print('---before training---')
    print(f'max_epoch = {max_epoch}')
    print(f'init_learning_rate = {learning_rate}')
    print(f'tau = {tau}\n')
    
    #early stopping的參數
    cnt = 0
    best_training_acc = 0
    best_testing_acc = 0
    best_w = np.array([0]*len(datas[0][0]), dtype = np.float64).transpose()
    best_b = 0
    
    #training的資料
    training_size = len(datas)
    testing_size = len(test)
    w = np.transpose(np.array([np.random.randn(len(datas[0][0]))], dtype = np.float64))
    b = np.random.randn()
    test_x, test_y = adjustment(test)
    train_x, train_y = adjustment(datas)
    y_head = 0
    #開始train
    epoch = 0
    for epoch in range(1, max_epoch + 1):
        #learning_rate每200次epoch除以1.5
        if epoch % 200 == 0:
            learning_rate /= 1.5
            
        #y_head = logistic(net_input)
        y_head = logistic(np.transpose(w) @ train_x + b)
        
        #當誤差足夠小時就停止程式
        if average_of_the_cross_entropy_loss(y_head, train_y) < tau:
            break
        
        dw = np.multiply(1 / training_size, (train_x @ np.transpose(y_head - train_y)))
        db = 1 / training_size * np.sum(y_head - train_y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        #計算training data的accuracy與testing data的accuracy
        training_acc = np.sum(((predict(logistic(np.transpose(w) @ train_x + b)) + 0) == train_y)) / training_size
        testing_acc = np.sum(((predict(logistic(np.transpose(w) @ test_x + b)) + 0) == test_y)) / testing_size
        
        #early stopping
        #當連續500個training_acc沒有比best_training_acc還好且連續500個best_testing_acc沒有比testing_ac還好
        #就直接回傳之前所記錄的最好的結果，反之則更新兩個的值
        if training_acc > best_training_acc and testing_acc > best_testing_acc:
            best_training_acc = training_acc
            best_testing_acc = testing_acc
            best_w = w
            best_b = b
            cnt = 0
        else:
            cnt += 1
        if cnt >= 500:
            print(f'stopped because it may not have better training results(early stopping)')
            print('---after training---')
            print(f'training acc = {best_training_acc * 100}%, testing acc = {best_testing_acc * 100}%')
            return learning_rate, epoch, best_w, best_b
        
    print('---after training---')
    print(f'training acc = {training_acc * 100}%, testing acc = {testing_acc * 100}%')
    if epoch == max_epoch:
        print('stopped because the epoch of training exceeds the maximum epoch we set(epoch >= max_epoch)')
    else:
        print('stopped because the error rate is less than tolerance(loss < tau)')
    return learning_rate, epoch, w, b

#輸出test.csv我們所預測的結果到test_ans.csv
def output_ans(w, b):
    #預測test.csv的分類
    data = pd.read_csv('test.csv').values
    ans = predict(logistic(data @ w + b))
    
    #將得到答案的1 or 0 轉成 2 or 5
    ans = np.array(list(map(trans2, ans))).reshape(-1, 1)
    
    #輸出結果到test_ans中
    pd.DataFrame(ans, columns = ['ans']).to_csv('test_ans.csv', index = False)


if __name__ == "__main__":
    #讀取資料
    dataset = read_training_data()
    
    #將十分之一的資料拿來做testing data
    test = dataset[3600:]
    train = dataset[:3600]
    
    # init_lr = 0.1, max_epoch = 2000, tau = 0.001
    #將參數(learing_rate, max_epoch, tau, training_data, testing_data)放入，並進行logistic regression
    lr,epoch, w, b = logistic_regression(0.1, 2000, 0.001, train, test)
    print(f'weight = {np.transpose(w)[0]}')
    print(f'bias = {b}')
    print(f'epoch = {epoch}')
    print(f'learning_rate = {lr}')
    output_ans(w, b)
