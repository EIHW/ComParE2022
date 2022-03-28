import numpy as np
import pandas as pd
from sklearn.metrics import recall_score


def CI(preds, labels):

    UARs = []
    for s in range(1000):
        np.random.seed(s)
        sample = np.random.choice(range(len(preds)), len(preds), replace=True) #boost with replacement
        sample_preds = preds[sample]
        sample_labels = labels[sample]

        uar = recall_score(sample_labels, sample_preds, average='macro')
        UARs.append(uar)

    q_0 = pd.DataFrame(np.array(UARs)).quantile(0.025)[0] #2.5% percentile
    q_1 = pd.DataFrame(np.array(UARs)).quantile(0.975)[0] #97.5% percentile

    return q_0, q_1