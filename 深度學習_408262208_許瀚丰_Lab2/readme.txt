因為在使用sigmoid function時1 / (1 + e^(-n))在numpy中會有overflow的問題，因此改正公式為0.5 * (1 + tanh(0.5 * n))
原sigmoid function如下
def sigmoid(n):
    return np.array(1 / (1 + np.exp(-n)), dtype = np.float64)
說明：
因tanh = (e^x - e^-x) / (e^x + e^-x)

0.5 * (1 + tanh(0.5 * x))
= 0.5 * (1 + (e^0.5x - e^-0.5x) / (e^0.5x + e^-0.5x)) 
= (e^0.5 / (e^0.5x + e^-0.5x))
= 1 / (1 + e^-x)


過程中我的learning rate有依照epoch次數慢慢下降，故一開始的learning rate與結束的learning rate會不同(一開始0.1，之後每200epoch時learning rate除以1.5)
