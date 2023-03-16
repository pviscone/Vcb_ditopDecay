# %%
#!------------------Imports-------------------


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score



# %%
#!------------------Data loading-------------------
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)


df = pd.read_pickle("../Jet_features.pkl", compression="bz2")

# [a,b]: a=non leptonic, b=leptonic
label = np.expand_dims(df["label"].astype(float).to_numpy(), axis=1)
ohe = OneHotEncoder()
ohe.fit(label)
label = ohe.transform(label).toarray()

data_df = df.loc[:, df.columns != "label"]
test_size = 0.5


event_id = data_df["event_id"]
data_df = data_df.loc[:, data_df.columns != "event_id"]
_, event_id_test, = train_test_split(
    event_id, test_size=test_size, shuffle=False)


train_data, test_data, train_label, test_label = train_test_split(
    data_df, label, test_size=test_size, shuffle=False)
#%%

sns.heatmap(data_df.corr(), vmin=-1, vmax=1, annot=False, cmap='viridis')

X_train = train_data.to_numpy()
y_train = train_label[:,1]
X_test = test_data.to_numpy()
y_test = test_label[:,1]

dt = DecisionTreeClassifier(max_depth=3,
                            min_samples_leaf=2)
bdt = AdaBoostClassifier(dt,
                         algorithm='SAMME',
                         n_estimators=8,
                         learning_rate=0.5)

bdt.fit(X_train, y_train)


y_predicted = bdt.predict(X_test)
print(classification_report(y_test, y_predicted,target_names=["background", "signal"]))
print(f"Area under ROC curve: {(roc_auc_score(y_test,bdt.decision_function(X_test)))}")


# %%

decisions = bdt.decision_function(X_test)
# Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(y_test, decisions)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid()
plt.show()
