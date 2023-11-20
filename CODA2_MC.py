########################
# Author：YZ Shi
# Date:   2022.8
# Loc.:   SZBL & WTU
# Description：Do Monte Carlo to predict RNA base-pairings 
#              using score matrix from CODA2 as constraint
# version: 1.0
########################
import sys, os
from collections import defaultdict  
from numpy import exp
import numpy  as np       
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import random
from math import sqrt
import time
t0 = time.time()
MC_step = 500000
T_list = [120.0,108.0,97.2,87.5,78.7,70.8,63.8,57.4,51.6,46.5,41.8,37.6,33.9,30.5,27.5,25.0] 
wt = 0.76  #0.9
U_lone = 0.8    # penalty of lone base pair
U_AU_end = 0.45  # penalty of AU at end
if len(sys.argv) < 2:
    print("USAGE: python CODA2_MC.py file_name path rand_seed (e.g, 5E54 ./ 0)")
    exit(1)
if len(sys.argv) == 2:
    Filepath = './'
    seed = 0
else:
    Filepath = sys.argv[2]
    seed = int(sys.argv[3])
score_path = Filepath + 'score/final_score.txt'
sequence_file = Filepath + 'data/' + sys.argv[1] + '.fasta'
#native_file = Filepath + 'data/' + sys.argv[1] + '-native.txt'
#out_file = 'MC/'+sys.argv[3] + '/'
#Outpath = Filepath + out_file
Outpath = Filepath + 'MC/'
if not os.path.exists(Outpath):
    os.makedirs(Outpath)
    print(f"folder '{Outpath}' is built")

basepair = ['AU','UA','GC','CG','GU','UG']
######### Read sequence ##########
def read_seq(file):
    from itertools import islice
    seq = []
    with open(file,'r') as f:
        for line in islice(f,1,None):
            seq = line.strip('\n')
    seqLen = len(seq)
    print(' sequence:',seq,'\n','lenSeq:',seqLen)
    return seq,seqLen
## Read the native base-base interaction derived from DSSR
## only used to evaluate the results
def native_bp(file):
    native = open(file,"r")
    lines = native.readlines()
    bplist = []
    for line in lines:   ##
        line = line.rstrip("\n") 
        sublist = line.split(" ")
        if int(sublist[0])<int(sublist[2]) and int(sublist[2])>0:
            bplist.append(sublist)
    #print("Number of native base pairs:", bplist.shape)
    return (bplist)
## Read the final_score calculated from CODA 
def Read_score(score_path,L):
    score_matrix = np.zeros((L,L),dtype=float) 
    score_list = []
    row = 0           
    with open(score_path,'r') as df:
        for line in df:
            score_list = line.strip('\n').split('\t') 
            score_matrix[row:] = score_list[0:L] 
            row+=1  
    val_max, val_min = np.nanmax(score_matrix), np.nanmin(score_matrix)
    print("Score max/mean/min: ",val_max,np.nanmean(score_matrix),val_min)  
    return score_matrix   

###based on pair info, get the stems
def pair2stem(pairs):    
    Stems = []
    edge = np.stack(pairs.nonzero())
    if edge.size > 0:        
        i_consistancy = (edge[0][1:] - edge[0][:-1])==1
        j_consistancy = (edge[1][1:] - edge[1][:-1])==-1
        consistancy = np.pad(i_consistancy & j_consistancy, (1,0), constant_values=True)
        segment = np.cumsum(~consistancy)
        N = len(segment)
        #print(N,"\n",segment)    
        for i in range(N):
            stem = (edge[:, segment==i].T)
            Stems.append(stem) 
    #print(Stems)
    return Stems
    
## Monte Carlo simulating Anneaning
def MC_SA(score_matrix,seq):   
    seqLen = len(seq)
    ## Thermodynamics parameters (Refs.: Turner et al. Biochemistry 1998, 37, 14719-14735 & J. Mol. Biol. (1999) 288, 911-940 & Biochemisty 2012, 3508-3522)
    '''
    stack = {'AAUU':-0.93,'UUAA':-0.93,'AUUA':-1.1,'UAAU':-1.33,'CUGA':-2.08,'AGUC':-2.08,
             'CAGU':-2.11,'UGAC':-2.11,'GUCA':-2.24,'ACUG':-2.24,'GACU':-2.35,'UCAG':-2.35,
             'CGGC':-2.36,'GGCC':-3.26,'CCGG':-3.26,'GCCG':-3.42,'AGUU':-0.35,'UUGA':-0.35,
             'AUUG':-0.90,'GUUA':-0.90,'CGGU':-1.25,'UGGC':-1.25,'CUGG':-1.77,'GGUC':-1.77,
             'GGCU':-1.80,'UCGG':-1.80,'GUCG':-2.15,'GCUG':-2.15,'GAUU':-0.51,'UUAG':-0.51,
             'GGUU':-0.25,'UUGG':-0.25,'GUUG':+0.72,'UGAU':-0.39,'UAGU':-0.39,'UGGU':-0.57}
    '''
    delt_H = {'AAUU':-6.82,'UUAA':-6.82,'AUUA':-9.38,'UAAU':-7.69,'CUGA':-10.48,'AGUC':-10.48,
              'CAGU':-10.44,'UGAC':-10.44,'GUCA':-11.40,'ACUG':-11.40,'GACU':-12.44,'UCAG':-12.44,
              'CGGC':-10.64,'GGCC':-13.39,'CCGG':-13.39,'GCCG':-14.88,'AGUU':-3.96,'UUGA':-3.96,
              'AUUG':-7.39,'GUUA':-7.39,'CGGU':-5.56,'UGGC':-5.56,'CUGG':-9.44,'GGUC':-9.44,
              'GGCU':-7.03,'UCGG':-7.03,'GUCG':-11.09,'GCUG':-11.09,'GAUU':-10.38,'UUAG':-10.38,
              'GGUU':-17.82,'UUGG':-17.82,'GUUG':-13.83,'UGAU':-0.96,'UAGU':-0.96,'UGGU':-12.64}
    delt_S = {'AAUU':-19.0,'UUAA':-19.0,'AUUA':-26.7,'UAAU':-20.5,'CUGA':-27.1,'AGUC':-27.1,
              'CAGU':-26.9,'UGAC':-26.9,'GUCA':-29.5,'ACUG':-29.5,'GACU':-32.5,'UCAG':-32.5,
              'CGGC':-26.7,'GGCC':-32.7,'CCGG':-32.7,'GCCG':-36.9,'AGUU':-11.6,'UUGA':-11.6,
              'AUUG':-21.0,'GUUA':-21.0,'CGGU':-13.9,'UGGC':-13.9,'CUGG':-24.7,'GGUC':-24.7,
              'GGCU':-16.8,'UCGG':-16.8,'GUCG':-28.8,'GCUG':-28.8,'GAUU':-31.8,'UUAG':-31.8,
              'GGUU':-56.7,'UUGG':-56.7,'GUUG':-46.9,'UGAU':-1.8,'UAGU':-1.8,'UGGU':-38.9}
    f1 = open(Outpath + 'mc_energy.txt', 'w')
    f2 = open(Outpath + 'secondary_structure.txt', 'w')
    f3 = open(Outpath + 'secondary_structure_minE.txt', 'w')
    pair = np.zeros([seqLen,seqLen],dtype=np.int32)
    base = [0 for i in range(seqLen)]
    U_min = 1000.0    
    U_before = 0.0    ###Energy Before Changes    
    tt = 0
    random.seed(seed) ##set random_seed for multi-MC & do MC simulations
    #print('random_seed:',seed,random.random())
    for T in T_list:
        TK = T+273.15
        D = 0.002*TK 
        for t in range(MC_step):          
            for i in range(100000):     ###Randomly choose two bases  
                i1 = random.randint(0,seqLen-5)        ##randomly select two bases
                i2 = random.randint(i1+4,seqLen-1)
                if (str(seq[i1])+str(seq[i2])) in basepair:
                    break       ##two bases must be possible paired type
            ###Changes the pairing bases
            pair_a = pair.copy()
            base_a = base.copy()
            stems_a = []
            if base[i1]==0 and base[i2]==0:   ##Two unpaired --> paired
                pair_a[i1,i2] = 1; base_a[i1] = 1; base_a[i2] = 1;
            elif base[i1]==0 and base[i2]==1: ##if i2 is paired with others, break the i2-j pair 
                pair_a[i1,i2] = 1; base_a[i1] = 1; base_a[i2] = 1;
                for j in range(seqLen):
                    if pair[i2,j]==1 or pair[j,i2]==1:
                         pair_a[i2,j] = 0; pair_a[j,i2] = 0; base_a[j] = 0;
                         break
            elif base[i1]==1 and base[i2]==0:
                pair_a[i1,i2] = 1; base_a[i1] = 1; base_a[i2] = 1;
                for j in range(seqLen):
                    if pair[i1,j]==1 or pair[j,i1]==1:
                         pair_a[i1,j] = 0; pair_a[j,i1] = 0; base_a[j] = 0;
                         break
            else:
                pair_a[i1,i2] = 1; base_a[i1] = 1; base_a[i2] = 1;  
                for j in range(seqLen):
                    if pair[i1,j]==1 or pair[j,i1]==1:
                         pair_a[i1,j] = 0; pair_a[j,i1] = 0; base_a[j] = 0;  
                    if pair[i2,j]==1 or pair[j,i2]==1:
                         pair_a[i2,j] = 0; pair_a[j,i2] = 0; base_a[j] = 0;                    
            ###Energy after Changes based on Stems
            U_after = 0.0
            stems_a = pair2stem(pair_a)
            for i in range(len(stems_a)):
                L=len(stems_a[i])            
                if L==1:
                    U_after += wt*U_lone
                elif L==2:
                    U_after += wt*U_lone*0.6 
                elif L==3: 
                    U_after += wt*U_lone*0.4 
                for j in range(L):
                    pair1=stems_a[i][j][0]
                    pair2=stems_a[i][j][1]
                    U_after += -score_matrix[pair1][pair2]
                    if j!=L-1:
                        pair3=stems_a[i][j+1][0] 
                        pair4=stems_a[i][j+1][1]
                        s1 = str(seq[pair1])+str(seq[pair3])+str(seq[pair2])+str(seq[pair4])
                        U_stack = delt_H[s1]-TK*0.001*delt_S[s1]
                        #U_stack = stack[s1]
                        if U_stack>0.0:
                            U_stack = 0.0
                        U_after += wt*U_stack 
                    if (j==0 or j==L-1):
                        if ((seq[pair1]=='U' and seq[pair2]=='A') or (seq[pair1]=='A' and seq[pair2]=='U')):
                            U_after += wt*U_AU_end
            ###Metroplis
            delt = U_after - U_before
            if delt <= 0:
                base=base_a; pair=pair_a; stems=stems_a;
                U_before=U_after
            else:
                p = random.random()
                if exp(-delt/D)>p:
                    base=base_a; pair=pair_a; stems=stems_a;
                    U_before=U_after
                else:
                    base=base; pair=pair; #stems=stems;
                    U_before=U_before 
            if U_before < U_min:
                U_min = U_before; pair_minE = pair.copy();
            if (t+1)%(MC_step*0.01)==0:
                tt += 1
                s2 = str(tt)+' '+str(T)+' '+str(t)+' '+str(round(U_before,3))+'\n'
                f1.write(s2)
                f1.flush()
            if (t+1)%(MC_step*0.5)==0:    
                print(T,' ', t,' ',U_before) 
    
        for i in range(seqLen):
            for j in range(seqLen):
                if pair[i,j]>0:
                   s3 = str(T)+' '+str(t)+' '+str(i)+' '+str(j)+' '+str(seq[i])+' '+str(seq[j])+'\n'
                   f2.write(s3)
                   f2.flush()            
    f1.close()
    f2.close()
    for i in range(seqLen):
        for j in range(seqLen):            
            if pair_minE[i,j]>0:
               s4 = str(i)+' '+str(j)+' '+str(seq[i])+' '+str(seq[j])+'\n'
               f3.write(s4)
               f3.flush()
    f3.close()
    ### Save the contact_map of conf with minE
    np.save(Outpath+"contact_minE.npy",pair_minE)
    ###Plot the energy vs MC_step
    energy = pd.read_table(Outpath + 'mc_energy.txt',sep =' ',header=None)
    energy.columns = ['t0','T','t','U']
    fig = plt.figure(figsize = (10,5),dpi = 80)
    plt.plot(energy['t0'],energy['U'])
    plt.xlabel('MC steps')
    plt.ylabel('Energy')
    plt.savefig(Outpath + 'MC_energy.jpg',dpi=100)
    #plt.show()
    return pair, pair_minE, U_min
###add line to one figure
def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, ls='--', c='black')
##Score_Heatmap plot
def plot_heatmap_score(name,mat,L,val_max,val_min=-10.0,ismask=0):
    mask = np.zeros((L, L))
    if ismask==1:
        for i in range(1,L):
            for j in range(i):
                if mat[i, j] != val_max:                
                    mask[i, j] = True
                if mat[j, i] != val_max:                
                    mask[j, i] = True
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
    plt.savefig(name +'.png', dpi=300)   

############ output the dbn format of 2D structure with minimum energy ##########
def pair2dbn(pair_matrix, seq):
    index = np.argwhere(pair_matrix == 1)
    pair_final = []
    pair_final = index[index[:,0]<index[:,1]]
    dbn = [0 for x in range(0, len(seq))]
    for index, value in enumerate(dbn):
        dbn[index] = "."
    for pair in pair_final:
        i, j = pair 
        if dbn[i] != "(" and dbn[j] != ")" and dbn[i] != "[" and dbn[j] != "]":
            if ")" in dbn[i:j] or "(" in dbn[i:j]:
               if "[" in dbn[i:j] or "]" in dbn[i:j]:
                   if "{" in dbn[i:j] or "}" in dbn[i:j]:
                       dbn[i] = "<"
                       dbn[j] = ">"
                   else:
                       dbn[i] = "{"
                       dbn[j] = "}"
               else:
                   dbn[i] = "["
                   dbn[j] = "]"
            else:
               dbn[i] = "("
               dbn[j] = ")"               
    dbn_str = "".join([x for x in dbn])
    2D_out = open(Outpath + '2D.dot', 'w')
    2D.out.write(dbn_str)
    2D.out.close()
    return dbn_str  
'''
## Evaluate the results
def Evaluation_score(pair,pair_minE,seq,U_min):
    seqLen = len(seq)
    pair_matrix = pair.copy()
    pair_matrix_minE = pair_minE.copy()
    bplist = native_bp(native_file)
    signal = []
    for i in range(1,seqLen):
        for j in range(i):
            for row in bplist:
                if int(row[2])>0 and (int(row[0])==j+1 and int(row[2])==i+1):
                    if (j,i) not in signal:
                        signal.append((j,i))   #
                        pair_matrix[i,j] = 1
                        pair_matrix_minE[i,j] = 1

    ## Evaluation contact map from low T
    Output_result = open(Outpath + 'MC_result.txt', 'w')
    s = '\t'+'    F1'+'\t'+'MCC'+'\t'+'precision'+'\t'+'TPR'+'\t'+'FPR'+'\t'+'TP'+'\t'+'FP'+'\t'+'FN'+'\t'+'TN'+'\n'
    Output_result.write(s)
    rightuptri = []
    for i in range(seqLen):
        for j in range(seqLen):
            if i<j:
                rightuptri.append((i,j))            
    TP = len([i for i in rightuptri if (pair_matrix[i[0],i[1]] == 1 and i in signal)])
    FN = len([i for i in rightuptri if ((i in signal) and (pair_matrix[i[0],i[1]] == 0))])
    FP = len([i for i in rightuptri if (pair_matrix[i[0],i[1]] == 1 and i not in signal)])
    TN = len([i for i in rightuptri if ((pair_matrix[i[0],i[1]] == 0) and i not in signal)])
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    precision = TP/(TP+FP)
    F1 = 2*TP/(2*TP+FN+FP)
    MCC = (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    print("Result for Lowest T:")
    print(' F1 = ',F1,'\n','MCC = ',MCC,'\n','precision = ',precision,'\n','TPR: ',TPR,'\t','FPR: ',FPR,'\n',TP,'\t',FP,'\n',FN,'\t',TN)
    s = 'Lowest T:'+'\t'+str(round(F1,3))+'\t'+str(round(MCC,3))+'\t'+str(round(precision,3))+'\t'+str(round(TPR,3))+'\t'+str(round(FPR,3))+'\t'+str(TP)+'\t'+str(FP)+'\t'+str(FN)+'\t'+str(TN)+'\n'   
    Output_result.write(s)
    ## Evaluation for contact map from minE
    rightuptri = []
    for i in range(seqLen):
        for j in range(seqLen):
            if i<j:
                rightuptri.append((i,j))            
    TP = len([i for i in rightuptri if (pair_matrix_minE[i[0],i[1]] == 1 and i in signal)])
    FN = len([i for i in rightuptri if ((i in signal) and (pair_matrix_minE[i[0],i[1]] == 0))])
    FP = len([i for i in rightuptri if (pair_matrix_minE[i[0],i[1]] == 1 and i not in signal)])
    TN = len([i for i in rightuptri if ((pair_matrix_minE[i[0],i[1]] == 0) and i not in signal)])
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    precision = TP/(TP+FP)
    F1 = 2*TP/(2*TP+FN+FP)
    print("Result for Lowest U")
    print('  F1 = ',F1,'\n','MCC = ',MCC,'\n','\n','precision = ',precision,'\n','TPR: ',TPR,'\t','FPR: ',FPR,'\n',TP,'\t',FP,'\n',FN,'\t',TN)
    s = 'Lowest U:'+'\t'+str(round(F1,3))+'\t'+str(round(MCC,3))+'\t'+str(round(precision,3))+'\t'+str(round(TPR,3))+'\t'+str(round(FPR,3))+'\t'+str(TP)+'\t'+str(FP)+'\t'+str(FN)+'\t'+str(TN)+'\n'
    Output_result.write(s)
    dbn = pair2dbn(pair_matrix_minE, seq)
    #print(dbn)
    s = 'final_2D:'+'\t'+str(round(U_min,2))+'\t'+dbn
    Output_result.write(s)
    Output_result.close()

    plot_heatmap_score(Outpath+"MC_minE_contac_map",pair_matrix_minE,seqLen,1,0,1) 
    plot_heatmap_score(Outpath+"MC_END_contac_map",pair_matrix,seqLen,1,0,1) 
'''

if __name__ == "__main__":

    sequence, seqLen = read_seq(sequence_file)
    score = Read_score(score_path,seqLen)
    pair, pair_minE, U_min = MC_SA(score,sequence)
    #Evaluation_score(pair,pair_minE,sequence,U_min)
    #plot_heatmap_score(Outpath+"MC_minE_contact_map",pair_minE,seqLen,1,0,1) 
    run_time = time.time() - t0
    print("The run time: ",run_time,"s")




