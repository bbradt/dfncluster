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
exclude = [
    'Nearest Neighbors',
    'Ada Boost',
    'Gradient Boost',
    'Bernoulli Naive Bayes',
    'Voting',
    'Gaussian Process',
    'Decision Tree',
    'Bagging'
]
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
mean_rows = []
std_rows = []

for clusterer, folder in CFOLDERS.items():
    directory = os.path.join(ROOT_DIR, folder)
    score_file = os.path.join(directory, 'scores.pkl')
    if not os.path.exists(score_file):
        print("The score file %s does not exist" % score_file)
        continue
    scores = pkl.load(open(score_file, 'rb'))
    test_cols = [cols for cols in scores.columns if 'test' in cols]
    test_scores = scores[test_cols].to_dict()
    mean_row = {"Clustering Algorithm": clusterer}
    std_row = {"Clustering Algorithm": clusterer}

    for method in test_cols:
        if method.replace('test', '').strip() in exclude:
            continue
        mean_row[method.replace('test', '').strip()] = np.mean(list(test_scores[method].values()))
        std_row[method.replace('test', '').strip()] = np.std(list(test_scores[method].values()))
        for k in test_scores[method].values():
            rows.append(dict(AUC=k, classifier=method.replace('test', '').strip(), clusterer=clusterer))
    mean_rows.append(mean_row)
    std_rows.append(std_row)
num_columns = len(test_cols)
num_rows = len(CLUSTERERS)
sb.set()

df = pd.DataFrame(rows)
mean_df = pd.DataFrame(mean_rows)
mean_df = mean_df.set_index('Clustering Algorithm')
mean_df = mean_df.round(decimals=3)
std_df = pd.DataFrame(std_rows)
std_df = std_df.round(decimals=3)
std_df = std_df.set_index('Clustering Algorithm')
std_df = std_df[mean_df.max().sort_values(ascending=False).index]
mean_df = mean_df[mean_df.max().sort_values(ascending=False).index]


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

new_std_rows = []

for i, row in mean_df.iterrows():
    new_std_row = {"Clustering Algorithm": i}
    for j, column in enumerate(mean_df.columns):
        stdget = std_df.loc[i][j]
        mn = row[column]
        maxmn = np.max(mean_df[column])
        std_string = str(row[column]) + " ± " + str(stdget)
        if mn == maxmn:
            std_string = "**%s**" % std_string
        elif mn + stdget >= maxmn:
            std_string = "*%s*" % std_string
        new_std_row[column] = std_string
    new_std_rows.append(new_std_row)

new_std_df = pd.DataFrame(new_std_rows)
new_std_df = new_std_df.set_index('Clustering Algorithm')
new_std_df = new_std_df[mean_df.max().sort_values(ascending=False).index]



plt.savefig(filename, bbox_inches='tight')
sb.boxplot(data=dfc, x='classifier', y='AUC', hue='classifier', ax=ax[-1])
axc = plt.gca()
figLegend = plt.figure()
plt.figlegend(*axc.get_legend_handles_labels(), loc='upper left')
figLegend.savefig('results/accuracy_legend.png', bbox_inches='tight')

print("Mean Results")
print(new_std_df)
if args.betas:
    mean_df.to_csv('results/%s_betas_mean_scores.csv' % (DATASET))
    new_std_df.to_csv('results/full_%s_betas_mean_scores.csv' % (DATASET))
else:
    mean_df.to_csv('results/%s_mean_scores.csv' % (DATASET))
    new_std_df.to_csv('results/full_%s_mean_scores.csv' % (DATASET))

#pp=sb.boxplot(data=dfc, x='classifier', y='AUC', hue='classifier')
# ax[-1].set_axis_off()
# pp.set_visible(False)
# plt.legend(framealpha=1.0, prop={'size': 24}, bbox_to_anchor=(1.5, 1))  # , loc=2, borderaxespad=0.)
