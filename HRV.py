import pandas as pd
import numpy as np
import math

def HRV_from_RR(arrayRR):

    sdnn = round(np.std(arrayRR), 2)
    NNdiff = np.abs(np.diff(arrayRR))
    rmssd = round(np.sqrt(np.sum((NNdiff * NNdiff) / len(NNdiff))), 2)
    sdnn = round(np.std(arrayRR), 2)
    HRV = np.log(rmssd)*100/6.5
#     print(rmssd)
    print(np.log(rmssd)*100)
    return rmssd, sdnn, HRV

# 2 minutes
interval = 2
def from_train_table_to_params(df, interval):
    
    convert_time = 1000*60
    
    stop_time = interval*convert_time
    
    x = np.cumsum(df['x'])
    df['timestamp'] = x
    
    a = df.loc[(df['timestamp'] <= (stop_time))]
    
    array_RR = np.array(a['x'])
#     print(array_RR)
    
    artifact_correction_threshold = 0.05
    filtered_RRs = []
    for i in range(len(array_RR)):
        if array_RR[(i-1)]*(1-artifact_correction_threshold) < array_RR[i] < array_RR[(i-1)]*(1+artifact_correction_threshold):
            filtered_RRs.append(array_RR[i])
#     print('filtr', len(filtered_RRs), len(array_RR))
    
    filtered_RRs = np.array(filtered_RRs)
    rmssd, sdnn, HRV = HRV_from_RR(filtered_RRs)
    
    return rmssd, sdnn, HRV
    
    