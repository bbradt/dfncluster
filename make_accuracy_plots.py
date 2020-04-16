import os
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="fbirn", type=str)
parser.add_argument("--betas", default=False, action="store_true")
args = parser.parse_args()
plt.close()
ROOT_DIR = 'results'
DATASET = args.dataset.lower()
CFOLDERS = dict(
    kmeans='kmeans_%s' % DATASET,
    gmm='gmm_%s' % DATASET,
    bgmm='bgmm_%s' % DATASET,
    dbscan='dbscan_%s' % DATASET,
    hierarchical='hierarchical_%s' % DATASET
)
if args.betas:
    for k, v in CFOLDERS.items():
        CFOLDERS[k] = v+"_betas"
CLUSTERERS = list(CFOLDERS.keys())
rows = []

for clusterer, folder in CFOLDERS.items():
    directory = os.path.join(ROOT_DIR, folder)
    score_file = os.path.join(directory, 'scores.pkl')
    if not os.path.exists(score_file):
        print("The score file %s does not exist" % score_file)
        continue
    scores = pkl.load(open(score_file, 'rb'))
    test_cols = [cols for cols in scores.columns if 'test' in cols]
    test_scores = scores[test_cols].to_dict()
    for method in test_cols:
        for k in test_scores[method].values():
            rows.append(dict(AUC=k, classifier=method.replace('test', '').strip(), clusterer=clusterer))

df = pd.DataFrame(rows)

num_columns = len(test_cols)
num_rows = len(CLUSTERERS)
sb.set()
fig, ax = plt.subplots(1, num_rows, figsize=(24, 12))
for i in range(num_rows):
    dfc = df[df['clusterer'] == CLUSTERERS[i]]
    # if i == num_rows - 1:
    #    sb.boxplot(data=dfc, x='classifier', y='AUC', hue='classifier', ax=ax[i])
    #    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # else:
    try:
        sb.boxplot(data=dfc, x='classifier',  y='AUC', ax=ax[i])
    except ValueError:
        continue
    ax[i].set_title(CLUSTERERS[i])
    ax[i].set_ylim([0.0, 1.0])
    ax[i].set_xticks(())
if args.betas:
    plt.suptitle('AUC using Beta-Features')
    filename = 'results/%s_betas_accuracy.png' % DATASET
else:
    plt.suptitle('AUC usign Cluster-Assignments')
    filename = 'results/%s_assignments_accuracy.png' % DATASET
plt.savefig(filename, bbox_inches='tight')
sb.boxplot(data=dfc, x='classifier', y='AUC', hue='classifier', ax=ax[-1])
axc = plt.gca()
figLegend = plt.figure()
plt.figlegend(*axc.get_legend_handles_labels(), loc='upper left')
figLegend.savefig('results/accuracy_legend.png', bbox_inches='tight')

#pp=sb.boxplot(data=dfc, x='classifier', y='AUC', hue='classifier')
# ax[-1].set_axis_off()
# pp.set_visible(False)
# plt.legend(framealpha=1.0, prop={'size': 24}, bbox_to_anchor=(1.5, 1))  # , loc=2, borderaxespad=0.)
