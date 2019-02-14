import sys
import csv
csv.field_size_limit(sys.maxsize)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import itertools
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,roc_curve
import scikitplot as skplot

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

if __name__=="__main__":
    match_m_df=pd.read_csv("charting-m-matches.csv", engine="python",error_bad_lines=False)
    POV_m_df=pd.read_csv("charting-m-stats-Overview.csv")
    match_m_df1=match_m_df[['match_id', 'Player 1', 'Player 2','Gender',
       'Date', 'Tournament', 'Round', 'Time','Surface','Best of']]
    POV_m_df1=POV_m_df[['match_id','player','set', 'aces', 'dfs', 'first_in',
       'first_won', 'second_in', 'second_won', 'bk_pts', 'bp_saved',
       'return_pts', 'return_pts_won', 'winners','unforced']]
    POV_m_df1=POV_m_df1.query("set=='Total'")
    POV_m_df2=POV_m_df1.groupby(["match_id"], as_index=False).agg("sum")
    POV_m_df2["total_points"]=POV_m_df2["return_pts"]
    POV_m_df2["serve_pts_won"]=POV_m_df2["first_won"]+POV_m_df2["second_won"]
    df=match_m_df1.merge(POV_m_df2, how='inner', on="match_id")
    print(df.head())
    df.iloc[448,8]="Clay"
    missing_data = df.isna()
    print(missing_data.describe())
    df["aces_p_set"]=df["aces"]/df["Best of"]
    df["dfs_p_set"]=df["dfs"]/df["Best of"]
    df["bp_p_set"]=df['bk_pts']/df["Best of"]
    df["bp_sav_p_set"]=df['bp_saved']/df["Best of"]
    df["winner_p_set"]=df['winners']/df["Best of"]
    df["ufe_p_set"]=df['unforced']/df["Best of"]
    df["fs_pct"]=df['first_in']/df['total_points']
    df["ss_pct"]=df['second_in']/df['total_points']
    df["spw_pct"]=df['serve_pts_won']/df['total_points']
    df["rpw_pct"]=df['return_pts_won']/df['total_points']
    # Clay - 1, Not Clay - 0
    df['Clay'] = df['Surface'].map({'Clay': 1, 'Hard': 0, "Grass":0})

    # Hard - 0, Not Hard - 1
    df['Hard'] = df['Surface'].map({'Clay': 0, 'Hard': 1, "Grass":0})

    # Grass- 1, Student Not Grass - 0
    df['Grass'] = df['Surface'].map({'Clay': 0, 'Hard': 0, "Grass":1})

    df.drop(['Time','Round','Tournament','player','aces', 'dfs',
       'first_in', 'first_won', 'second_in', 'second_won', 'bk_pts',
       'bp_saved', 'return_pts', 'return_pts_won','winners', 'unforced'],axis=1,inplace=True)

    y=df[["Clay"]]
    x=df.drop(["Clay"],axis=1)
    print(y.head())
    print(x.head())

    X_train, X_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.3)
    X_train.hist(bins=20,figsize=(15,15),grid=False)

    plt.figure(figsize=(12,3))
    plt.subplot(1, 3, 1)
    plt.hist(X_train["winner_p_set"],bins=20,color="yellowgreen")
    plt.xlabel("Winners/Set")

    plt.subplot(1,3,2)
    plt.hist(X_train["fs_pct"],bins=20,color="greenyellow")
    plt.xlabel("First Serve %")

    plt.subplot(1,3,3)
    plt.hist(X_train["rpw_pct"],bins=20,color="yellowgreen")
    plt.xlabel("Return Pts Won %")

    plt.savefig("images/final_hists")

    scatter_info=X_train[[#'total_points', #'serve_pts_won',
        'aces_p_set', 'dfs_p_set',
       'bp_p_set', 'bp_sav_p_set', 'winner_p_set', 'ufe_p_set', 'fs_pct',
       'ss_pct', 'spw_pct', 'rpw_pct']]
    pd.scatter_matrix(scatter_info, figsize=(20, 20),grid=False)
    plt.savefig("images/scatter_mat")

    bar_info1 = pd.DataFrame({'lab':['Hard', 'Clay', 'Grass'], 'val':[len(X_train[["Hard"]].query("Hard==1")),len(y_train[["Clay"]].query("Clay==1")),len(X_train[["Grass"]].query("Grass==1"))]})
    bar_info1.plot.bar(x='lab', y='val', rot=0,legend=False, title="Court Type Balance",fontsize=10,color=["deepskyblue","salmon","mediumspringgreen"])

    bar_info2 = pd.DataFrame({'lab1':["Not_Clay","Clay"], 'val1':[len(y_train[["Clay"]].query("Clay==0")),len(y_train[["Clay"]].query("Clay==1"))]})
    bar_info2.plot.bar(x='lab1', y='val1', rot=0,legend=False, title="Clay/Not Clay Balance",fontsize=10,color=["lightseagreen","salmon"])

    X_mod1 = X_train[['aces_p_set', 'dfs_p_set','bp_p_set', 'winner_p_set', 'ufe_p_set', 'fs_pct', 'rpw_pct']].values
    y_mod1 = y_train[["Clay"]].values
    y_mod1=y_mod1.ravel()

    logit_model1=LogisticRegression().fit(X_mod1, y_mod1)
    print(f"Model 1 cofficients: {logit_model1.coef_}")

    kfold1 = KFold(len(y_train))

    accuracies1 = []
    precisions1 = []
    recalls1 = []

    for train_index, test_index in kfold1:
        model = LogisticRegression()
        model.fit(X_mod1[train_index], y_mod1[train_index])
        y_predict = model.predict(X_mod1[test_index])
        y_true = y_mod1[test_index]
        accuracies1.append(accuracy_score(y_true, y_predict))
        precisions1.append(precision_score(y_true, y_predict))
        recalls1.append(recall_score(y_true, y_predict))

    print ("accuracy mod1:", np.average(accuracies1))
    print ("precision mod1:", np.average(precisions1))
    print ("recall mod1:", np.average(recalls1))

    X_mod2 = X_train[['aces_p_set', 'dfs_p_set', 'winner_p_set', 'fs_pct', 'rpw_pct']].values
    y_mod2 = y_train[["Clay"]].values

    logit_model2 = LogisticRegression().fit(X_mod2,y_mod2.ravel())
    print(f"Model 1 cofficients: {logit_model2.coef_}")

    kfold2 = KFold(len(y_train))

    accuracies2 = []
    precisions2 = []
    recalls2 = []

    for train_index, test_index in kfold2:
        model = LogisticRegression()
        model.fit(X_mod2[train_index], y_mod2[train_index])
        y_predict = model.predict(X_mod2[test_index])
        y_true = y_mod2[test_index]
        accuracies2.append(accuracy_score(y_true, y_predict))
        precisions2.append(precision_score(y_true, y_predict))
        recalls2.append(recall_score(y_true, y_predict))

    print ("accuracy mod2:", np.average(accuracies2))
    print ("precision mod2:", np.average(precisions2))
    print ("recall mod2:", np.average(recalls2))

    X_mod3 = X_train[['aces_p_set', 'dfs_p_set', 'winner_p_set', 'fs_pct', 'rpw_pct']].values
    y_mod3 = y_train[["Clay"]].values

    logit_model3 = LogisticRegression(class_weight='balanced').fit(X_mod3,y_mod3.ravel())
    print(f"Model 1 cofficients: {logit_model3.coef_}")

    kfold3 = KFold(len(y_train))

    accuracies3 = []
    precisions3 = []
    recalls3 = []

    for train_index, test_index in kfold3:
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_mod3[train_index], y_mod3[train_index])
        y_predict = model.predict(X_mod3[test_index])
        y_true = y_mod3[test_index]
        accuracies3.append(accuracy_score(y_true, y_predict))
        precisions3.append(precision_score(y_true, y_predict))
        recalls3.append(recall_score(y_true, y_predict))

    print ("accuracy mod3:", np.average(accuracies3))
    print ("precision mod3:", np.average(precisions3))
    print ("recall mod3:", np.average(recalls3))

    y_predict = logit_model3.predict(X_test[['aces_p_set', 'dfs_p_set', 'winner_p_set', 'fs_pct', 'rpw_pct']])
    y_true = y_test
    print(f"TEST SET ACCURACY: {accuracy_score(y_true, y_predict)}")
    print(f"TEST SET PRECISION: {precision_score(y_true, y_predict)}")
    print(f"TEST SET RECALL:{recall_score(y_true, y_predict)}")

    plot_confusion_matrix(confusion_matrix(y_true, y_predict),["Clay","Not Clay"])

    y_hats_test=logit_model3.predict_proba(X_test[['aces_p_set', 'dfs_p_set', 'winner_p_set', 'fs_pct', 'rpw_pct']].values)
    skplot.metrics.plot_roc_curve(y_test, y_hats_test, curves='macro')
    plt.savefig("images/roc")
