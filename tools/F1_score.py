import os
import glob
import pandas as pd
import numpy as np
import argparse
import math

from torch.nn.functional import threshold

def main_topk(path, dataset, annotation, version):
    
    files_tmp = os.listdir(path)
    files = sorted(files_tmp, key = lambda x:int(x[-2:]))
    ann_csv = pd.read_csv(annotation)
    test_path_temp = [os.path.join(path, i, 'test_detection') for i in files]
    txts = glob.glob(os.path.join(test_path_temp[0], '*.txt'))

    txts = [int(i.split('_')[-1].split('.')[0]) for i in txts]
    txts.sort()
    # txt_index = txts[-1]
    best, best_m1, best_m2 = 0, 0, 0
    if dataset =='cas(me)^2':
        out_path_tmp = os.path.join(os.path.dirname(annotation), 'top_k', 'catop_k'+'_'+str(version))
    else:
        out_path_tmp = os.path.join(os.path.dirname(annotation), 'top_k', 'satop_k'+'_'+str(version))
    if not os.path.exists(out_path_tmp):
        os.makedirs(out_path_tmp)
    best_out = os.path.join(out_path_tmp, 'best_sample.log')
    topk_out = os.path.join(out_path_tmp, 'topk.log')  
    if os.path.exists(topk_out):
        os.remove(topk_out)
    for e in range(1, len(txts)):
        txt_index = txts[e]
        test_path = [os.path.join(i, 'test_'+str(txt_index).zfill(2)+'.txt') for i in test_path_temp]
        print('number of epochs:',txt_index)
        # confirm the best top_k
        for k in range(2, 30):
            standard_out = os.path.join(out_path_tmp, 'epoch'+str(e)+'_'+str(k)+'_'+'sample.log')
            FP, FN, TP = 0, 0, 0
            TP1, TP2 = 0, 0
            N1, N2, N_all = 0, 0, 0
            write_list = list()
            length_count = list()
            for i in test_path:
                with open(i, 'r') as f:
                    all_lines = f.readlines()
                all_lines = [h.split('\t') for h in all_lines]
                
                # divide predicitons of every video
                count = 1
                tmp_list = list()
                all_test = dict()
                # no duplicate label extraction
                all_video = list(set([name[0] for name in all_lines]))
                # number of GT of every video
                num_of_video = len(all_test.keys()) 

                for tv in all_video:
                    tmp_video = tv
                    for j in range(len(all_lines)):
                        if all_lines[j][0] == tmp_video:
                            tmp_list.append(all_lines[j])
                    all_test[count] = tmp_list
                    count = count + 1
                    tmp_list = list()
                # least len of GT
                # len_pre = [len(i) for i in all_test.values()]
                # least_len = min(len_pre)
                part_pre = [i[:k] for i in all_test.values()]

                # predictions: sorted by strat bondaries
                part_pre= [sorted(i, key = lambda x:int(float(x[1]))) for i in part_pre]
                
                # N1: number of precictions of macro-expressions
                # N2: number of precictions of micro-expressions
                # N_all: number of precictions
                N_all = N_all + len(part_pre) * k
                for part in part_pre:
                    N1 = N1 + len([o for o in part if o[-2] == '1'])
                    N2 = N2 + len([o for o in part if o[-2] == '2'])

                f_out = open(standard_out, 'a')
                for video_num, pre in enumerate(part_pre):
                    video_name_list = list(set(ann_csv.video.values[:].tolist()))
                    video_name_list.sort()
                    
                    video_name_last = part_pre[video_num][0][0]
                    video_name_part = 's' + video_name_last[:2]
                    video_name = os.path.join(video_name_list[0].split('/s')[0], video_name_part, video_name_last)

                    video_ann_df = ann_csv[ann_csv.video == video_name]
                    act_start_video = video_ann_df['startFrame'].values[:]
                    # act_start_video = [sorted(i, key = lambda x:int(x)) for i in act_start_video]
                    
                    indexes = np.argsort(act_start_video)
                    act_end_video = video_ann_df['endFrame'].values[:]
                    act_end_video = np.array(act_end_video)[indexes]

                    labels = video_ann_df['type_idx'].values[:]
                    labels = np.array(labels)[indexes]
                    
                    act_start_video.sort()
                    # act_end_video = video_ann_df['endFrame'].values[:]
                    # act_end_video.sort()
                    # act_end_video = [sorted(i, key = lambda x:int(x)) for i in act_end_video]
                    
                    # calculate f1-score
                    # number of actual frames that have been calculated so far
                    act_num = 0
                    for m in range(len(act_start_video)):
                        # number of predictions that have been calculated so far
                        count_num = 0
                        sign = False
                        for n in range(len(pre)):
                            # if the dataset is samm, predictions should be increased by a factor of label_frequency
                            pre_start = int(int(float(pre[n][1])) * label_frequency)
                            pre_end = int(int(float(pre[n][2])) * label_frequency)
                            act_start = int(act_start_video[m])
                            act_end = int(act_end_video[m])
                            
                            if act_start > pre_end:
                                # prevent duplicate records
                                if write_list and (pre_start in write_list[-1] and pre_end in write_list[-1]):
                                    # if there is no remaining predicted samples, record gts directly
                                    if n==len(pre)-1:    
                                        for l in range(m,len(act_start_video)):
                                            act_num  = act_num + 1
                                            FN = FN + 1
                                            tmp_write_list = [video_name_last[:7], int(act_start_video[l]), int(act_end_video[l]), '_', '_', 'FN']
                                            write_list.append(tmp_write_list)  
                                        break                                 
                                    else:
                                        continue
                                # if there is no associated gts, record predicted samples directly
                                else:
                                    tmp_write_list = [video_name_last[:7], '_', '_', pre_start, pre_end, 'FP']
                                    if tmp_write_list not in write_list:
                                        FP = FP + 1
                                        write_list.append(tmp_write_list)   

                                if n==len(pre)-1:    
                                    for l in range(m,len(act_start_video)):
                                        act_num  = act_num + 1
                                        FN = FN + 1
                                        tmp_write_list = [video_name_last[:7], int(act_start_video[l]), int(act_end_video[l]), '_', '_', 'FN']
                                        write_list.append(tmp_write_list)  
                                    break
                            # record gts by calculating IoUs & analysing labels
                            elif  act_start<pre_end and pre_start<act_end:
                                iou = (min(pre_end, act_end) - max(pre_start, act_start))/(max(pre_end, act_end) - min(pre_start, act_start))
                                if iou >= 0.5 and labels[m]==int(float(pre[n][3])):
                                    TP = TP + 1
                                    if labels[m] == 1:
                                        TP1 = TP1 + 1
                                    else:
                                        TP2 = TP2 + 1
                                    tmp_write_list = [video_name_last[:7], act_start,act_end, pre_start, pre_end, 'TP']
                                    write_list.append(tmp_write_list)
                                    length_count.append(act_end-act_start)
                                else:
                                    FP = FP + 1
                                    tmp_write_list = [video_name_last[:7], act_start,act_end, pre_start, pre_end, 'FP']
                                    write_list.append(tmp_write_list)
                                sign = True
                            elif act_end < pre_start:
                                if count_num == 0 and pre_start != pre_end:
                                    FN = FN + 1
                                    tmp_write_list = [video_name_last[:7], act_start, act_end, '_', '_', 'FN']
                                    write_list.append(tmp_write_list)
                                    if m==len(act_start_video)-1:    
                                        for l in range(len(pre)):
                                            FP = FP + 1
                                            tmp_write_list = [video_name_last[:7], '_', '_', int(float(pre[l][1])), int(float(pre[l][2])), 'FP']
                                            write_list.append(tmp_write_list)
                                        break
                                    else:
                                        pre = pre[count_num:]
                                        break
                                elif pre_start == pre_end:
                                    FN = FN + 1
                                    tmp_write_list = [video_name_last[:7], act_start, act_end, '_', '_', 'FN']
                                    write_list.append(tmp_write_list)
                                    break
                                else:
                                    if m==len(act_start_video)-1: 
                                        if not sign: 
                                            FN = FN + 1
                                            tmp_write_list = [video_name_last[:7], act_start, act_end, '_', '_', 'FN']
                                            write_list.append(tmp_write_list)  
                                        pre = pre[count_num:]
                                        for l in range(len(pre)):
                                            FP = FP + 1
                                            tmp_write_list = [video_name_last[:7], '_', '_', int(float(pre[l][1])), int(float(pre[l][2])), 'FP']
                                            write_list.append(tmp_write_list)
                                    else:
                                        if not sign: 
                                            FN = FN + 1
                                            tmp_write_list = [video_name_last[:7], act_start, act_end, '_', '_', 'FN']
                                            write_list.append(tmp_write_list)  
                                        pre = pre[count_num:]
                                    break
                            count_num = count_num + 1
                        if act_num==len(act_start_video):
                            break
                        act_num = act_num +1
 
            # calculate F1_score
            # M_all need to calculate in SAMM
            # M1： Number of macro-expressions
            # M2： Number of micro-expressions
            if dataset == 'cas(me)^2':
                M1= 282
                M2 = 54
            else:
                M1 = 340
                M2 = 159
            recall1 = 1.0* TP1/M1
            recall2 =1.0* TP2/M2
            recall_all = 1.0 *(TP1+TP2)/(M1+M2)
            # Sometimes, there are no predictions of micro-expressions or macro-expressions
            if TP1==0 and TP2 !=0:
                precision1 = 0
                precision2 = 1.0* TP2/N2
                precision_all = 1.0* (TP1+TP2)/(N1+N2)
                F1_SCORE_M1 = 0
                F1_SCORE_M2 = 2*(recall2*precision2)/(recall2+precision2)
                F1_SCORE = 2*(recall_all*precision_all)/(recall_all+precision_all)
            elif TP1!=0 and TP2 ==0: 
                precision1 = 1.0* TP1/N1 
                precision2 = 0
                precision_all = 1.0* (TP1+TP2)/(N1+N2)
                F1_SCORE_M1 = 2*(recall1*precision1)/(recall1+precision1)
                F1_SCORE_M2 = 0
                F1_SCORE = 2*(recall_all*precision_all)/(recall_all+precision_all)
            elif TP1==0 and TP2 ==0:
                precision1 = 0
                precision2 = 0
                precision_all = 0
                F1_SCORE_M1 = 0
                F1_SCORE_M2 = 0
                F1_SCORE = 0
            else:
                precision1 = 1.0* TP1/N1
                precision2 = 1.0* TP2/N2
                precision_all = 1.0* (TP1+TP2)/(N1+N2)
                F1_SCORE_M1 = 2*(recall1*precision1)/(recall1+precision1)
                F1_SCORE_M2 = 2*(recall2*precision2)/(recall2+precision2)
                F1_SCORE = 2*(recall_all*precision_all)/(recall_all+precision_all)


            if F1_SCORE_M1 > best_m1:
                best_m1 = F1_SCORE_M1
                print("f1_score_macro: %05f,  f1_score_micro: %05f"%(best_m1, best_m2))
            if F1_SCORE_M2 > best_m2:
                best_m2 = F1_SCORE_M2
                print("f1_score_macro: %05f,  f1_score_micro: %05f"%(best_m1, best_m2))
            # record best the F1_scroe and the result of predictions
            if F1_SCORE > best:
                best = F1_SCORE
                print('number of epoch: %d, topk: %5f'%(e, k))
                print("recall: %05f,  precision: %05f,  f1_score: %05f"%(recall_all, precision_all, best))
                with open(best_out, 'w') as f_sout:
                    f_sout.writelines("%s, %s, %s, %s, %s, %s\n" % (wtmp[0], wtmp[1],wtmp[2],wtmp[3],wtmp[4],wtmp[5]) for wtmp in write_list)
                if F1_SCORE > 0.25:
                    standard_out = os.path.join(out_path_tmp, str(e)+'_'+str(k)+'_'+'sample.log')
                    with open(standard_out, 'w') as f_sout:
                        f_sout.writelines("%s, %s, %s, %s, %s, %s\n" % (wtmp[0], wtmp[1],wtmp[2],wtmp[3],wtmp[4],wtmp[5]) for wtmp in write_list)
                    with open(topk_out, 'a') as f_threshold:
                        f_threshold.writelines("%d, %f, %d, %d, %d, %f\n" % (e, k, TP, FP, FN, F1_SCORE))
                length_count.sort()
                print(length_count, TP, TP1, TP2, N1, N2)
        

def main_threshold(path, dataset, annotation, version, label_frequency, start_threshold):
    
    files_tmp = os.listdir(path)
    files = sorted(files_tmp, key = lambda x:int(x[-2:]))
    ann_csv = pd.read_csv(annotation)
    test_path_temp = [os.path.join(path, i, 'test_detection') for i in files]
    txts = glob.glob(os.path.join(test_path_temp[0], '*.txt'))

    txts = [int(i.split('_')[-1].split('.')[0]) for i in txts]
    txts.sort()

    best, best_m1, best_m2 = 0, 0, 0

    if dataset =='cas(me)^2':
        out_path_tmp = os.path.join(os.path.dirname(annotation), 'threshold', 'cathreshold'+'_'+str(version))
    else:
        out_path_tmp = os.path.join(os.path.dirname(annotation), 'threshold', 'sathreshold'+'_'+str(version))
    if not os.path.exists(out_path_tmp):
        os.makedirs(out_path_tmp)
    best_out = os.path.join(out_path_tmp, 'best_sample.log')
    threshold_out = os.path.join(out_path_tmp, 'threshlod.log')
    if os.path.exists(threshold_out):
        os.remove(threshold_out)
    for e in range(4, 30):
        txt_index = txts[e]
        test_path = [os.path.join(i, 'test_'+str(txt_index).zfill(2)+'.txt') for i in test_path_temp]    
        # confirm the best threshold
        for k_temp in range(start_threshold, 900, 1):
            k = 1.0 *k_temp/1000  
            FP, FN, TP = 0, 0, 0
            TP1, TP2 = 0, 0
            N1, N2, N_all = 0, 0, 0
            length_count = list()
            write_list = list()
            for i in test_path:
                with open(i, 'r') as f:
                    all_lines = f.readlines()
                all_lines = [h.split('\t') for h in all_lines]
                
                # divide all gts of every video
                tmp_video = all_lines[0][0]
                count = 1
                tmp_list = list()
                all_test = dict()
                all_video = list(set([name[0] for name in all_lines]))
                for tv in all_video:
                    tmp_video = tv
                    for j in range(len(all_lines)):
                        if all_lines[j][0] == tmp_video:
                            tmp_list.append(all_lines[j])
                    all_test[count] = tmp_list
                    count = count + 1
                    tmp_list = list()
                # number of GT of every video
                num_of_video = len(all_test.keys()) 
                # least len of GT
                part_tmp = list()
                
                # select predictions of every video (prob > threshold)
                for i in range(num_of_video):
                    tmp_one_video = list(all_test.values())[i]
                    part = [o for o in tmp_one_video if float(o[-1][:-2]) > k ]
                    # N1: number of precictions of macro-expressions
                    # N2: number of precictions of micro-expressions
                    # N_all: number of precictions
                    N_all = N_all + len(part)
                    N1 = N1 + len([o for o in part if o[-2] == '1'])
                    N2 = N2 + len([o for o in part if o[-2] == '2'])
                    if not part:
                        part = [[tmp_one_video[0][0], '100000', '100000', '_','_']]
                    part_tmp.append(part)   
                part_pre = part_tmp
               
                # predictions: sorted by prob
                part_pre= [sorted(i, key = lambda x:int(float(x[1]))) for i in part_pre]

                for video_num, pre in enumerate(part_pre):
                    video_name_list = list(set(ann_csv.video.values[:].tolist()))
                    video_name_list.sort()
                    
                    # identify the current subject
                    video_name_last = part_pre[video_num][0][0]
                    if dataset =='cas(me)^2':
                        video_name_part = 's' + video_name_last[:2]
                        video_name = os.path.join(video_name_list[0].split('/s')[0], video_name_part, video_name_last)
                    else:
                        video_name = os.path.join(video_name_list[0][:-4],str(video_name_last).zfill(3))
                    
                    # select startframes of every video
                    video_ann_df = ann_csv[ann_csv.video == video_name]
                    act_start_video = video_ann_df['startFrame'].values[:]
                    # select indexes of startframes of every video
                    indexes = np.argsort(act_start_video)
                    # labels and endframes are sorted by indexes from actual start frames
                    act_end_video = video_ann_df['endFrame'].values[:]
                    act_end_video = np.array(act_end_video)[indexes]
                    labels = video_ann_df['type_idx'].values[:]
                    labels = np.array(labels)[indexes]
                    # actual start frames are sorted by time series
                    act_start_video.sort()
                    
                    # calculate f1-score
                    # number of actual frames that have been calculated so far
                    act_num = 0
                    for m in range(len(act_start_video)):
                        # number of predictions that have been calculated so far
                        count_num = 0
                        sign = False
                        for n in range(len(pre)):
                            # if the dataset is samm, predictions should be increased by a factor of label_frequency
                            pre_start = int(int(float(pre[n][1])) * label_frequency)
                            pre_end = int(int(float(pre[n][2])) * label_frequency)
                            act_start = int(act_start_video[m])
                            act_end = int(act_end_video[m])
                            
                            if act_start > pre_end:
                                # prevent duplicate records
                                if write_list and (pre_start in write_list[-1] and pre_end in write_list[-1]):
                                    # if there is no remaining predicted samples, record gts directly
                                    if n==len(pre)-1:    
                                        for l in range(m,len(act_start_video)):
                                            act_num  = act_num + 1
                                            FN = FN + 1
                                            tmp_write_list = [video_name_last[:7], int(act_start_video[l]), int(act_end_video[l]), '_', '_', 'FN']
                                            write_list.append(tmp_write_list)  
                                        break                                 
                                    else:
                                        continue
                                # if there is no associated gts, record predicted samples directly
                                else:
                                    tmp_write_list = [video_name_last[:7], '_', '_', pre_start, pre_end, 'FP']
                                    if tmp_write_list not in write_list:
                                        FP = FP + 1
                                        write_list.append(tmp_write_list)   

                                if n==len(pre)-1:    
                                    for l in range(m,len(act_start_video)):
                                        act_num  = act_num + 1
                                        FN = FN + 1
                                        tmp_write_list = [video_name_last[:7], int(act_start_video[l]), int(act_end_video[l]), '_', '_', 'FN']
                                        write_list.append(tmp_write_list)  
                                    break
                            # record gts by calculating IoUs & analysing labels
                            elif  act_start<pre_end and pre_start<act_end:
                                iou = (min(pre_end, act_end) - max(pre_start, act_start))/(max(pre_end, act_end) - min(pre_start, act_start))
                                if iou >= 0.5 and labels[m]==int(float(pre[n][3])):
                                    TP = TP + 1
                                    if labels[m] == 1:
                                        TP1 = TP1 + 1
                                    else:
                                        TP2 = TP2 + 1
                                    tmp_write_list = [video_name_last[:7], act_start,act_end, pre_start, pre_end, 'TP']
                                    write_list.append(tmp_write_list)
                                    length_count.append(act_end-act_start)
                                else:
                                    FP = FP + 1
                                    tmp_write_list = [video_name_last[:7], act_start,act_end, pre_start, pre_end, 'FP']
                                    write_list.append(tmp_write_list)
                                sign = True
                            elif act_end < pre_start:
                                if count_num == 0 and pre_start != pre_end:
                                    FN = FN + 1
                                    tmp_write_list = [video_name_last[:7], act_start, act_end, '_', '_', 'FN']
                                    write_list.append(tmp_write_list)
                                    if m==len(act_start_video)-1:    
                                        for l in range(len(pre)):
                                            FP = FP + 1
                                            tmp_write_list = [video_name_last[:7], '_', '_', int(float(pre[l][1])), int(float(pre[l][2])), 'FP']
                                            write_list.append(tmp_write_list)
                                        break
                                    else:
                                        pre = pre[count_num:]
                                        break
                                elif pre_start == pre_end:
                                    FN = FN + 1
                                    tmp_write_list = [video_name_last[:7], act_start, act_end, '_', '_', 'FN']
                                    write_list.append(tmp_write_list)
                                    break
                                else:
                                    if m==len(act_start_video)-1: 
                                        if not sign: 
                                            FN = FN + 1
                                            tmp_write_list = [video_name_last[:7], act_start, act_end, '_', '_', 'FN']
                                            write_list.append(tmp_write_list)  
                                        pre = pre[count_num:]
                                        for l in range(len(pre)):
                                            FP = FP + 1
                                            tmp_write_list = [video_name_last[:7], '_', '_', int(float(pre[l][1])), int(float(pre[l][2])), 'FP']
                                            write_list.append(tmp_write_list)
                                    else:
                                        if not sign: 
                                            FN = FN + 1
                                            tmp_write_list = [video_name_last[:7], act_start, act_end, '_', '_', 'FN']
                                            write_list.append(tmp_write_list)  
                                        pre = pre[count_num:]
                                    break
                            count_num = count_num + 1
                        if act_num==len(act_start_video):
                            break
                        act_num = act_num +1
            
            # calculate F1_score
            # M_all need to calculate in SAMM
            # M1： Number of macro-expressions
            # M2： Number of micro-expressions
            if dataset == 'cas(me)^2':
                M1= 282
                M2 = 54
            else:
                M1 = 340
                M2 = 159
            recall1 = 1.0* TP1/M1
            recall2 =1.0* TP2/M2
            recall_all = 1.0 *(TP1+TP2)/(M1+M2)
            # Sometimes, there are no predictions of micro-expressions or macro-expressions
            if TP1==0 and TP2 !=0:
                precision1 = 0
                precision2 = 1.0* TP2/N2
                precision_all = 1.0* (TP1+TP2)/(N1+N2)
                F1_SCORE_M1 = 0
                F1_SCORE_M2 = 2*(recall2*precision2)/(recall2+precision2)
                F1_SCORE = 2*(recall_all*precision_all)/(recall_all+precision_all)
            elif TP1!=0 and TP2 ==0: 
                precision1 = 1.0* TP1/N1 
                precision2 = 0
                precision_all = 1.0* (TP1+TP2)/(N1+N2)
                F1_SCORE_M1 = 2*(recall1*precision1)/(recall1+precision1)
                F1_SCORE_M2 = 0
                F1_SCORE = 2*(recall_all*precision_all)/(recall_all+precision_all)
            elif TP1==0 and TP2 ==0:
                precision1 = 0
                precision2 = 0
                precision_all = 0
                F1_SCORE_M1 = 0
                F1_SCORE_M2 = 0
                F1_SCORE = 0
            else:
                precision1 = 1.0* TP1/N1
                precision2 = 1.0* TP2/N2
                precision_all = 1.0* (TP1+TP2)/(N1+N2)
                F1_SCORE_M1 = 2*(recall1*precision1)/(recall1+precision1)
                F1_SCORE_M2 = 2*(recall2*precision2)/(recall2+precision2)
                F1_SCORE = 2*(recall_all*precision_all)/(recall_all+precision_all)

            if F1_SCORE_M1 > best_m1:
                best_m1 = F1_SCORE_M1
                print("f1_score_macro: %05f,  f1_score_micro: %05f"%(best_m1, best_m2))
            if F1_SCORE_M2 > best_m2:
                best_m2 = F1_SCORE_M2
                print("f1_score_macro: %05f,  f1_score_micro: %05f"%(best_m1, best_m2))
            # record best the F1_scroe and the result of predictions
            if F1_SCORE > best:
                best = F1_SCORE
                print('number of epoch: %d, threshold: %5f'%(e, k))
                print("recall: %05f,  precision: %05f,  f1_score: %05f"%(recall_all, precision_all, best))
                with open(best_out, 'w') as f_sout:
                    f_sout.writelines("%s, %s, %s, %s, %s, %s\n" % (wtmp[0], wtmp[1],wtmp[2],wtmp[3],wtmp[4],wtmp[5]) for wtmp in write_list)
                if F1_SCORE > 0.25:
                    standard_out = os.path.join(out_path_tmp, str(e)+'_'+str(k)+'_'+'sample.log')
                    with open(standard_out, 'w') as f_sout:
                        f_sout.writelines("%s, %s, %s, %s, %s, %s\n" % (wtmp[0], wtmp[1],wtmp[2],wtmp[3],wtmp[4],wtmp[5]) for wtmp in write_list)
                    with open(threshold_out, 'a') as f_threshold:
                        f_threshold.writelines("%d, %f, %d, %d, %d, %f\n" % (e, k, TP, FP, FN, F1_SCORE))
                length_count.sort()
                print(length_count, TP, TP1, TP2, N1, N2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test')

    # parser.add_argument('--path', type=str, default=r'/home/yww/mer_spot/output_V28/cas(me)^2')
    # parser.add_argument('--ann', type=str, default=r'/home/yww/mer_spot/casme2_annotation.csv')
    # parser.add_argument('--dataset', type=str, default=r'cas(me)^2')
    # parser.add_argument('--version', type=int, default=28)
    # parser.add_argument('--top_k', type=bool, default=False)
    # parser.add_argument('--label_frequency', type=float, default=1.0)
    # parser.add_argument('--start_threshold', type=int, default=300)

    parser.add_argument('--path', type=str, default=r'/home/yww/mer_spot/output_V28/samm_5_5')
    parser.add_argument('--ann', type=str, default=r'/home/yww/mer_spot/samm_annotation_merge.csv')
    parser.add_argument('--dataset', type=str, default=r'samm')
    parser.add_argument('--version', type=int, default=28)
    parser.add_argument('--top_k', type=bool, default=False)
    parser.add_argument('--label_frequency', type=float, default=5.0)
    parser.add_argument('--start_threshold', type=int, default=100)
    args = parser.parse_args()

    path = args.path
    dataset = args.dataset
    ann = args.ann
    version = args.version
    top_k = args.top_k
    label_frequency = args.label_frequency
    start_threshold = args.start_threshold
    if top_k:
        main_topk(path, dataset, ann, version)
    else:
        main_threshold(path, dataset, ann, version, label_frequency, start_threshold)