import os
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
plt.close()
seed = 12345
ROOT_DIR = 'results'
CFOLDERS = dict(
    kmeans='GaussKmeans-%s' % seed,
    gmm='GaussGMM-%s' % seed,
    bgmm='GaussBGMM-%s' % seed,
    dbscan='GaussDBSCAN-%s' % seed
)
CLUSTERERS = list(CFOLDERS.keys())
rows = []

for clusterer, folder in CFOLDERS.items():
    directory = os.path.join(ROOT_DIR, folder)
    scores = pkl.load(open(os.path.join(directory, 'scores.pkl'), 'rb'))
    test_cols = [cols for cols in scores.columns if 'test' in cols]
    test_scores = scores[test_cols].to_dict()
    for method in test_cols:
        for k in test_scores[method].values():
            rows.append(dict(AUC=k, classifier=method.replace('test', '').strip(), clusterer=clusterer))

df = pd.DataFrame(rows)

num_columns = len(test_cols)
num_rows = len(CLUSTERERS)
sb.set()
fig, ax = plt.subplots(1, num_rows+1)
for i in range(num_rows):
    dfc = df[df['clusterer'] == CLUSTERERS[i]]
    # if i == num_rows - 1:
    #    sb.boxplot(data=dfc, x='classifier', y='AUC', hue='classifier', ax=ax[i])
    #    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # else:
    sb.boxplot(data=dfc, x='classifier',  y='AUC', ax=ax[i])
    ax[i].set_title(CLUSTERERS[i])
    ax[i].set_ylim([0.3, 1.0])
#sb.boxplot(data=dfc, x='classifier', y='auc', hue='classifier', ax=ax[-1])
sb.boxplot(data=dfc, x='classifier', y='AUC', hue='classifier', ax=ax[-1])
ax[-1].set_axis_off()
plt.legend(framealpha=1.0, prop={'size': 18})  # , bbox_to_anchor=(0, 0), loc=2, borderaxespad=0.)
plt.show()