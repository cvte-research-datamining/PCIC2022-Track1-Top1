import numpy as np
import pandas as pd
import os


def main():
    load_path = '../'
    path = '../../../../results/CVTEDMer_1/'
    if not os.path.exists(path):
        os.makedirs(path)

    prob1 = pd.read_csv(load_path + 'lstm/prob.csv')
    prob2 = pd.read_csv(load_path + 'conv/prob.csv')
    prob3 = pd.read_csv(load_path + 'finetune/prob.csv')

    prob1['prob'] = 0.8 * prob1['prob_mean'] + 0.15 * prob2['prob_mean'] + 0.05 * prob3['prob_mean']
    prob1['rank'] = prob1['prob'].rank()
    preds_all_threshold = np.where(prob1['rank'] > (len(prob1) - 6387), 1, 0)
    sub = pd.DataFrame({'label': preds_all_threshold})
    sub[['label']].to_csv(path + 'submission.csv', index=False, header=False)


if __name__ == '__main__':
    main()
