import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
import numpy

#'./datasets/creditcard.csv'
def ReadCSV (path, mark):
    data= pd.read_csv(path)
    labels = data[mark]
    data = data.drop(columns=[mark], axis=1)
    return data, labels


def MinMaxTransform(df):
    scaler=MinMaxScaler()
    scaler.fit(df)
    df=scaler.transform(df)
    return df


#def CreateMatrix(vector):
''' 
len_=10
vector=np.array([1,2,4,5,7,5])
matrix =[np.copy(vector)]
matrix[0]=vector
for i in range(len_):
    copy = np.copy(vector)
    np.random.shuffle(copy)
    matrix.append(copy)
    print(i)
   # matrix[i]=np.random.shuffle(vector)
print(np.array(matrix)) '''

def get_threshold_PR(predict, labels_test, threshold_step=0.01, end_step=1, step_width=1):
    #thresholds = [i * threshold_step for i in range(1,int(1 / threshold_step),step_width)]
    thresholds = [i * threshold_step for i in range(1, int(end_step / threshold_step), step_width)]
    samples = pd.DataFrame(columns=('True Fraud', 'False Fraud', 'FraudRate', 'FraudRecall','FPR','f1_score'))
    for threshold in thresholds:
        print(threshold)
        if threshold >= end_step: break
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in enumerate(labels_test):
            if predict[i[0]] < threshold and i[1] == 0:
                TN += 1
            elif predict[i[0]] < threshold and i[1] == 1:
                FN += 1
            elif predict[i[0]] >= threshold and i[1] == 1:
                TP += 1
            else:
                FP += 1
        precision=round(TP / ((TP + FP) if (TP + FP) > 0 else 1),3 )
        recall=round(TP / ((TP + FN) if (TP + FN) > 0 else 1),3 )
        fpr= round((FP/(FP+TN)if (FP+TN) > 0 else 1),3)
        samples = pd.concat([samples, pd.DataFrame({
                                                    'Threshold':threshold,
                                                    'True Fraud': TP, 'False Fraud': FP,
                                                    'FraudRate': precision,
                                                    'FraudRecall': recall,
                                                    'FPR':fpr,
                                                    'f1_score': 2*precision*recall/(precision+recall)
                                                    if(precision+recall)>0 else 1 },
                                                   index=[0])],
                            axis=0, ignore_index=True)
    return samples