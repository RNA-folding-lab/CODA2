########################
# Author：YZ Shi, Z Zhang, D Luo, YQ Huang
# Date:   2022.8 (last update at 2023.7)
# Loc.:   SZBL & WTU
# Description：inferring base-pairings from deep mutational scanning and mobility-based selection data
########################
import sys, os
from collections import defaultdict
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
import seaborn as sns
from math import sqrt,exp,log
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import time
t0 = time.time()
## Set file_path & parameters & read input files (double reads, native contact, and sequecne)
if len(sys.argv)<2:
    print("USAGE: python CODA2_scoring.py &file_name &file_path (e.g, 5TPY ./)")
    exit(1)
if len(sys.argv)==2:
    Filepath = './' 
else:    
    Filepath = sys.argv[2] + '/'
mut_file = Filepath + 'data/' + sys.argv[1] + '.var.ra'
sequence_file = Filepath + 'data/' + sys.argv[1] + '.fasta'
#native_file = Filepath + 'data/' + sys.argv[1] + '-native.txt'
Outpath = Filepath + 'score/'
if not os.path.exists(Outpath):
    os.makedirs(Outpath)
    print(f"folder '{Outpath}' is built")
####difine the hyper-parameters
C_value, gamma_value, sd_cut = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
#C_value = 0.01
#gamma_value = 0.1          #para C & gamma of SVR
lower_output_cut = 5.0        #delete the sample with output/input<lower_cut
#sd_cut = 3.5                 #delt_fitness > mean+sd_cut*sd --> possible pairing 
######### Read sequence ##########
def read_seq(file):
    from itertools import islice
    seq = []
    with open(file,'r') as f:
        for line in islice(f,1,None):
            seq = line.strip('\n')
    seqLen = len(seq)
    return seq,seqLen
###add line to one figure
def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, ls='--', c='black')
##Score_Heatmap plot
def plot_heatmap_score(name,mat,L,val_max,val_min=-5.0,ismask=0):
    mask = np.zeros((L, L))
    if ismask==1:
        for i in range(1,L):
            for j in range(i):
                if mat[i, j] != val_max:                
                    mask[i, j] = True
    #mask[np.tril_indices_from(mask)] = True  #set the below as 1
    plt.figure(figsize=(7, 5))
    #sns.set(font_scale=1.5)
    ax = sns.heatmap(mat,mask=mask,cmap="YlGn",vmin=val_min,vmax=val_max)   
    abline(1,0) 
    plt.xlim(0,L)
    plt.ylim(L,0)
    plt.xticks(rotation=90)
    #plt.yticks(range(0,L,5))
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)
    plt.savefig(name +'.png', dpi=800)   
##Get the wide_type seq. for each position 
def WT_AA(num):
    return sequence[num]
##Read mutation file (from Zhang Z) & output the single/double/triple mutation data
def read_mut_data(file):
    f=open(file,'r+')
    single=open(Outpath + 'single.txt', 'w')
    double=open(Outpath + 'double.txt', 'w')
    #triple=open(Outpath + 'triple.txt', 'w')
    for line in f:
        spt=line.strip().split('\t')
        mutNum=int(spt[0])
        if mutNum==0:
            Cwt=float(spt[1])
        elif mutNum==1:
            single.write(spt[1]+"\t"+spt[2]+"\t"+spt[4]+"\t"+spt[5]+"\t"+spt[6]+"\n")
        elif mutNum==2:
            double.write(spt[1]+"\t"+spt[2]+"\t"+spt[3]+"\t"+spt[4]+"\t"+spt[5]+"\t"+spt[6]+"\t"+spt[8]+"\t"+spt[9]+"\t"+spt[10]+"\n")
        #elif mutNum==3:
            #triple.write(spt[1]+"\t"+spt[2]+"\t"+spt[3]+"\t"+spt[4]+"\t"+spt[5]+"\t"+spt[6]+"\t"+spt[7]+"\t"+spt[8]+"\t"+spt[9]+"\t"+spt[12]+"\n")
        else:
            continue    
    return Cwt   
def Read_mut_dataframe(file):
    Cwt = read_mut_data(file)  
    ############ Re-read mutation data ##########
    in_single=Outpath+'single.txt'
    single=pd.read_table(in_single,sep ='\t',header = None,names = ['Pos','Mut','input','output','fitness'])
    in_double=Outpath+'double.txt'
    double=pd.read_table(in_double,sep ='\t',header = None,names = ['Pos1','Pos2','Mut1','Mut2','input','output',
                                                                'fitness1','fitness2','fitness'])
    #in_triple=Outpath+'triple.txt'
    #triple=pd.read_table(in_triple,sep ='\t',header = None,names = ['Pos1','Pos2','Pos3','Mut','input','output',
                                                                #'fitness1','fitness2','fitness3','fitness'])
    #print("single/double/triple shape: ",single.shape,double.shape,triple.shape)
    ####put the native seq into the double data
    double['wt1'] = double.apply(lambda x:WT_AA(x['Pos1']),axis=1)
    double['wt2'] = double.apply(lambda x:WT_AA(x['Pos2']),axis=1) 
    ## sort the file keeping Pos from small to big and Pos1<Pos2
    double = double.sort_values(['Pos1','Pos2'],ascending = [True,True])
    if min(double['Pos1'])!=0:   ##keep the first pos is 0
        double['Pos1'] = double['Pos1']-1
        double['Pos2'] = double['Pos2']-1
    double = double.reset_index(drop=True)
    print("Mut_data read successfully, & Cwt = ",Cwt)
    return Cwt,single,double #,triple

########## one hot encode for sequences ##########
def one_hot_encode(sequences):
    sequences_arry = np.array(list(sequences)).reshape(-1, 1)
    lable = np.array(list('ACGT')).reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(lable)
    seq_encode = enc.transform(sequences_arry).toarray()
    return (seq_encode)
####### Just consider base pairs: AT/TA/GC/CG/GT/TG
def bpcheck(n1,n2):
    if ((n1=='G' and n2=='C') or (n1=='C' and n2=='G') or (n1=='A' and n2=='T') 
        or (n1=='T' and n2=='A') or (n1=='G' and n2=='T') or (n1=='T' and n2=='G')):
        return True
    else:
        return False
##### generate gaussian pdf
def normfun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

#### Filter the double data
# delete outliers based on boxplot of input/output
def box_outlier(data):
    df = data.copy(deep=True)
    print("Before filter data: ",df.shape)
    out_index = []
    for col in df.columns[[4,5]]:     # input/output/fitness1/fitness2/fitness  
        Q1 = df[col].quantile(q=0.25)       # lower quartile
        Q3 = df[col].quantile(q=0.75)       # upper quartile
        #low_whisker = Q1 - 1.5 * (Q3 - Q1)  # lower edge
        low_whisker = 5
        up_whisker = Q3 + 25 * (Q3 - Q1)   # upper edge
        #print(low_whisker,up_whisker)
        rule = (df[col] > up_whisker) | (df[col] < low_whisker) 
        out = df[col].index[rule]
        out_index += out.tolist()  
    df.drop(out_index, inplace=True)
    df = df[(df.output > lower_output_cut) & (df.input > lower_output_cut)]
    df = df.reset_index(drop=True)
    print("After filter: ",df.shape)
    return df
###Just consider AU, GU & GC pairings
def Keep_WC_pairing(df):
    df = df[((df.wt1=='A')&(df.wt2=='U')) | ((df.wt1=='G')&(df.wt2=='C')) | 
            ((df.wt1=='C')&(df.wt2=='G')) | ((df.wt1=='U')&(df.wt2=='A')) | 
            ((df.wt1=='U')&(df.wt2=='G')) | ((df.wt1=='G')&(df.wt2=='U'))]
    df = df[((df.Mut1=='A')&(df.Mut2=='T'))|((df.Mut1=='G')&(df.Mut2=='C'))|
            ((df.Mut1=='C')&(df.Mut2=='G'))|((df.Mut1=='T')&(df.Mut2=='A'))|
            ((df.Mut1=='T')&(df.Mut2=='G')) | ((df.Mut1=='G')&(df.Mut2=='T'))]
    return df
####Preprocess initial fitness data
def Filter_data(data):
    df = data.copy(deep=True)
    print("Before filter data: ",df.shape)
    df = Keep_WC_pairing(df)
    #df.drop(df[(df['fitness']>df['fitness1']) & (df['fitness']>df['fitness2'])].index,inplace=True)
    #df.drop(df[df['fitness']<0.01].index,inplace=True)
    df.drop(df[(df['fitness']>5*(df['fitness1']+df['fitness2']))].index,inplace=True)
    print("After filter: ",df.shape)
    df = df.reset_index(drop=True)
    return df
###Build the X,Y dataset for SVR regression
def encode_data(data):
    tempdata=one_hot_encode(data)
    temp_dataframe=pd.DataFrame(tempdata,columns=list('acgt'))
    return (temp_dataframe)
def XY_dataset(data):
    df = data.copy(deep=True)
    ## Build training set: I-fitness~fitness1+fitness2; 
    #X = df.loc[:,['fitness1','fitness2']]
    #Y = df.loc[:,['fitness']]

    ## Build training set: II-fitness~fitness1+fitness2+seq_encode+position
    x_temp = df.loc[:,['Pos1','Pos2','fitness1','fitness2']]
    Y = df.loc[:,['fitness']]
    en_mut1=encode_data(df['Mut1'])  #one-hot for sequence
    en_mut2=encode_data(df['Mut2'])
    en_wt1=encode_data(df['wt1'])
    en_wt2=encode_data(df['wt2'])
    X_temp=pd.concat([x_temp,en_mut1,en_mut2,en_wt2,en_wt2],axis=1)
    standard_data=MinMaxScaler().fit_transform(X_temp)
    X=pd.DataFrame(standard_data)
    return X,Y
## using SVR() --> fitness~fitness1+fitness2 
def svr_model_fit(x,y):    
    from sklearn.svm import SVR        
    svr_rbf = SVR(kernel='rbf', C=C_value, gamma=gamma_value)
    gs = svr_rbf.fit(x,y.values.ravel())
    return svr_rbf
##Training the regression model (SVR or MLP)
def Training_regression_model(data):
    df0 = data.copy(deep=True)
    df1 = box_outlier(data)    
    X1,Y1 = XY_dataset(df1)
    svr = svr_model_fit(X1,Y1)
    
    df0 = Filter_data(df0)
    X0,Y0 = XY_dataset(df0)
    y_pred = svr.predict(X0)
    #y_pred = svr.predict(X1)

    #calculate the delt between pred. & real
    df = df0.copy(deep=True)
    df['real'] = Y0.round(3)
    df['pred'] = y_pred.round(3)
    df['delt'] = (df['real'] - df['pred']).round(3)
    df['delt_m'] = (df['delt']/(df['pred']+0.2)).round(3)

    #plot the relation or distribution of pred & real fitness
    fig = plt.figure(figsize = (15,5))
    plt.subplot(1,3,1)
    plt.plot(df['real'],df['pred'],'o',markersize=1)
    #plt.xlim(0,2.0)
    #plt.ylim(0,2.0)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    abline(1,0)
    plt.subplot(1,3,2)
    plt.plot(df['real'],label="real")
    plt.plot(df['pred'], label="predict")
    plt.legend()
    plt.xlabel('Site')
    plt.ylabel('Fitness')
    plt.subplot(1,3,3)
    df.delt_m.plot(kind='hist',bins=50,color='blue',density=True)
    df.delt_m.plot(kind='kde',color='red')
    plt.xlabel('delt')
    plt.tight_layout()
    plt.savefig(Outpath + 'distribution_delt_fitness_svr.png',dpi=600)
    ## output the file including pred_fitness & delt
    df2 = df.loc[:,['Pos1','Pos2','wt1','wt2','Mut1','Mut2','delt','delt_m']]
    df2.to_csv(Outpath + 'pred_fitness.txt',sep = '\t', index = False)
    return df2

##Calculate P(fitness) based on two Gaussian distributions
def DeltToScore(ra, mean1, sd1, mean2, sd2,p1):
    p_unpaired = normfun(ra, mean1, sd1)
    p_paired = normfun(ra, mean2, sd2)
    ##p1 = 0.67/(L-1)   #used in CODA1 by Z. Zhang
    return log(p_paired/(p_paired*p1 + p_unpaired*(1-p1)))
##output the score matrix
def Score_matrix(deltList,posList,L,listA,listB):
    paired_prob = len(listB)/len(listA)
    print ("p(paired): "+str(paired_prob)+"   lenA: "+str(len(listA))+"   lenB:"+str(len(listB))) 
    meanA,sdA = np.mean(listA), np.std(listA)
    meanB,sdB = np.mean(listB), np.std(listB)
    print('meanA/sdA:',meanA,sdA,'\n','meanB/sdB:',meanB,sdB)
    score_matrix = [[np.NAN for i in range(L)] for j in range(L)]
    for i in range(len(deltList)):
        final_score = 0
        for j in range(len(deltList[i])):
            final_score += DeltToScore(deltList[i][j],meanA, sdA, meanB, sdB,paired_prob)
        final_score /= len(deltList[i])
        #final_score = DeltToScore(np.nanmean(deltList[i]),meanA, sdA, meanB, sdB,paired_prob)
        score_matrix[posList[i][0]][posList[i][1]] = final_score
        score_matrix[posList[i][1]][posList[i][0]] = final_score
    #print(score_matrix)
    f = open(Outpath + 'initial_score.txt', 'w')
    for i in range(L):
        score_list = []
        for j in range(L):
            score_list.append(str(round(score_matrix[i][j], 2)))
        f.write('\t'.join(score_list)+'\n')
    f.close()
    val_max, val_min = np.nanmax(score_matrix), np.nanmin(score_matrix)
    plot_heatmap_score(Outpath+"Score_map",score_matrix,L,val_max,val_min)
    return score_matrix
## Plot delt distribution
'''
def Plot_delt_distribution(deltList_temp,listA,listB,mean0,sd0):
    meanA,sdA = np.mean(listA), np.std(listA)
    meanB,sdB = np.mean(listB), np.std(listB)
    ###ditribution of classified score
    fig = plt.figure(figsize = (10,5))
    plt.subplot(1,2,1)
    plt.hist(deltList_temp,color='blue',density=True)
    x = np.arange(min(deltList_temp),max(deltList_temp),0.1)
    y1 = normfun(x, mean0, sd0)
    plt.plot(x,y1,color='red',linewidth = 3,label="all")
    plt.xlabel('Delt_score')
    plt.legend()
    plt.subplot(1,2,2)
    plt.hist(listA,color='blue',density=True)
    y2 = normfun(x, meanA, sdA)
    plt.plot(x,y2,linewidth = 2,label="listA")
    plt.hist(listB,color='green',density=True)
    y3 = normfun(x, meanB, sdB)
    plt.plot(x,y3,linewidth = 2,label="listB")
    plt.xlabel('Delt_score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Outpath + 'distribution_delt_score.jpg',dpi=600)
'''
## Calculate the score for each position (i,j):  (sum or max delt_m) 
def Contact_prediction(data,L):
    df = data.copy(deep=True)
    df = Training_regression_model(df)
    Output_Score = open(Outpath + 'score.txt', 'w')
    subarray = np.zeros((L,L))
    deltList_temp, deltList, posList = [], [], []
    for i in range(L-1):
        for j in range(i+3,L):
             sub_df = df.loc[(df['Pos1'] == i)&(df['Pos2'] == j),['delt_m']]
             score_temp = []
             if sub_df.empty == False:            
                 subarray[i,j] = round(sum(sub_df['delt_m']),3)   ##take sum of the score with AT/GC/GU at each pos1-pos2
                 #subarray[i,j] = round(max(sub_df['delt_m']),3)  ##take max value as score
                 score_temp = list(sub_df['delt_m'])
                 deltList.append(score_temp)
                 posList.append((i,j))
             for k in range(len(score_temp)):
                 deltList_temp.append(score_temp[k])
                 s = str(i)+'\t'+str(j)+'\t'+str(subarray[i,j])+'\t'+str(score_temp[k])+'\n'
                 Output_Score.write(s)
    Output_Score.close()
    mean0 = np.mean(deltList_temp)
    sd0 = np.std(deltList_temp)
    #print("deltList_l/mean/sd: ",len(deltList_temp),mean0,sd0)

    ##group the samples (ListA & ListB) according to whether score < mean0+?*sd0
    listA, listB = [], []
    for i in range(len(deltList_temp)):
        if deltList_temp[i] < mean0 + sd_cut*sd0:  #3sd0,B->31个；2sd0，B->89; sd0 ->339
             listA.append(deltList_temp[i])
        else:        
             listB.append(deltList_temp[i])    
    if len(listB)<10:
        print("ERROR: sd_cut could be too large")
        exit(1)
    #Plot_delt_distribution(deltList_temp,listA,listB,mean0,sd0)
    val_max = np.nanmax(subarray)
    #plot_heatmap_score(Outpath+"Score_native_map_delt",subarray,L,val_max,-2,1)
    score_matrix = Score_matrix(deltList,posList,L,listA,listB)
    return score_matrix
'''
## Read the native base-base interaction derived from DSSR
## and evaluate the results
def native_bp(file):
    native = open(file,"r")
    lines = native.readlines()
    bplist = []
    for line in lines:   ##
        line = line.rstrip("\n") 
        sublist = line.split(" ")
        if int(sublist[0])<int(sublist[2]) and int(sublist[2])>0:
            bplist.append(sublist)
    #print(bplist)
    return (bplist)
## Evaluation the results
def Evaluation_score(native_file,score_matrix,L):
    bplist = native_bp(native_file)
    #print(bplist)
    ##put the native interaction into the score_matrix and output
    val_max = np.nanmax(score_matrix)
    val_min = np.nanmin(score_matrix)
    print("val_max/min",val_max,val_min)
    signal = []
    for i in range(1,L):
        for j in range(i):
            score_matrix[i][j] = np.NAN
            for row in bplist:
                if int(row[2])>0 and (int(row[0])==j+1 and int(row[2])==i+1):
                    if (j,i) not in signal:
                        signal.append((j,i))   #
                        score_matrix[i][j] = val_max
    ## Evaluation
    rightuptri = []
    for i in range(L):
        for j in range(L):
            if i<j:
                rightuptri.append((i,j))            
    thlst = np.arange(val_min,val_max,0.1).tolist()
    #print(len(thlst))
    tprlst, fprlst, pcslst, F1lst, MCClst = [],[],[],[],[]
    for threshold in thlst:
        TP = len([i for i in rightuptri if (score_matrix[i[0]][i[1]] >= threshold and i in signal)])
        FN = len([i for i in rightuptri if ((i in signal) and (score_matrix[i[0]][i[1]] <= threshold))])
        FP = len([i for i in rightuptri if (score_matrix[i[0]][i[1]] >= threshold and i not in signal)])
        TN = len([i for i in rightuptri if ((score_matrix[i[0]][i[1]] <= threshold) and i not in signal)])
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        precision = TP/(TP+FP)
        F1 = 2*TP/(2*TP+FN+FP)
        MCC = (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        tprlst.append(TPR)
        fprlst.append(FPR)
        pcslst.append(precision)
        F1lst.append(F1)
        MCClst.append(MCC)
    evadf = pd.DataFrame({'threshold':thlst,'TPR(recall)':tprlst,'FPR':fprlst,'precision':pcslst,'F1':F1lst,'MCC':MCClst})
    Max_F1 = max(evadf['F1'])
    Max_MCC = max(evadf['MCC'])
    print("Max_F1:",Max_F1,"MAX_MCC",Max_MCC)
    ########plot the results
    fig = plt.figure(figsize = (15,5))
    plt.subplot(131)
    plt.plot(fprlst, tprlst)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.subplot(132)
    plt.plot(tprlst, pcslst)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.subplot(133)
    plt.plot(thlst, F1lst)
    plt.xlim(-10,10)
    plt.xlabel('Threshold')
    plt.ylabel('F1 score')
    plt.tight_layout()
    plt.savefig(Outpath + 'Evaluate_score.jpg',dpi=600)
    #plt.show()
    plot_heatmap_score(Outpath+"Score_native_map",score_matrix,L,val_max)
    return score_matrix,Max_F1,Max_MCC 

'''
## Min-Max normalize each row, respectively
def row_normalize(matrix_data):
    newdata = np.zeros(np.shape(matrix_data)) 
    Zmax,Zmin=matrix_data.max(axis=1),matrix_data.min(axis=1)
    row,col = matrix_data.shape[0],matrix_data.shape[1]
    for i in range(row):
        for j in range(col):
            newdata[i][j] = (matrix_data[i][j]-Zmin[i])/(Zmax[i]-Zmin[i]+0.00001)
    return newdata
## Min-Max normalize each col, respectively
def col_normalize(data):
    Zmax,Zmin=data.max(axis=0),data.min(axis=0)
    data = (data-Zmin)/(Zmax-Zmin)
    return data
## Read the final_score and preprocessing 
def Final_score(L,val=-6.0):
    score_matrix = np.zeros((L,L),dtype=float) 
    score_list = []
    row = 0           
    score_path = Outpath + 'initial_score.txt'
    with open(score_path,'r') as df:
        for line in df:
            score_list = line.strip('\n').split('\t') 
            score_matrix[row:] = score_list[0:L] 
            row+=1  
    print("Score max/mean/min: ",np.nanmax(score_matrix),np.nanmean(score_matrix),np.nanmin(score_matrix)) 
    ## Preprocess the score: <0-->0 & normalized
    for i in range(L):
        for j in range(L):
            if score_matrix[i][j]<=val or np.isnan(score_matrix[i][j]):   ## ?
                score_matrix[i][j]=val 
    score_matrix = row_normalize(score_matrix)
    #print("Score max/mean/min after process: ",np.nanmax(score_matrix),np.nanmean(score_matrix),np.nanmin(score_matrix)) 
    f = open(Outpath + 'final_score.txt', 'w')
    for i in range(L):
        score_list = []
        for j in range(L):
            score_list.append(str(round(score_matrix[i][j], 2)))
        f.write('\t'.join(score_list)+'\n')
    f.close()


if __name__ == "__main__":

    sequence, seqLen = read_seq(sequence_file)
    print(' sequence:',sequence,'\n','seqLen:',seqLen)
    Cwt,single,double = Read_mut_dataframe(mut_file)
    score_matrix = Contact_prediction(double,seqLen)
    #Int_score_matrix,Max_F1,Max_MCC = Evaluation_score(native_file,score_matrix,seqLen)
    Final_score(seqLen)
    run_time = time.time() - t0
    print("The run time: ",run_time,"s")
