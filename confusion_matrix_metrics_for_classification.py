"""
=============================================================================
Filename:       Confusion_matrix_metrics_for_classification.py
Author :        Jerome Gransac
Created:        2020-01-13
Last modified:  2020-01-23
Modified by:    Jerome Gransac
EDIT Jerome Gransac - 23/01/2020 : Removing overall confidence from the file 

Description:    Calculate confusion matrix metrics for each trained catgeory of a classification project
How to use:     Edit the variables in the file Confusion_matrix_metrics_for_classification_config

				confusion_matrix_metrics_for_classification.py

What that does : 
    It calculates ML metrics based on confusion matrix
    from https://en.wikipedia.org/wiki/Precision_and_recall

    Metrics are: confidence (avg prob) + recall, selectivity, precision, NPV, FNR, FTR, accurcy, F1
    Formulas:
    sensitivity, recall, hit rate, or true positive rate (TPR)
    TPR = TP / (TP + FN)
    specificity, selectivity or true negative rate (TNR)
    TNR = TN / (TN + FP)
    precision or positive predictive value (PPV)
    PPV = TP / (TP + FP)
    negative predictive value (NPV)
    NPV = TN / (TN + FN)
    accuracy (ACC)
    ACC= (TP + TN) / (TP + TN + FP + FN)
    F1 score is the harmonic mean of precision and sensitivity
    F1 = 2 * (PPV * TPR) / (PPV + TPR)

    miss rate or false negative rate (FNR) - NOT IMPLEMENTED
    FNR = FN / (FN + TP)
    fall-out or false positive rate (FPR) - NOT IMPLEMENTED
    FPR = FP / (FP + TN)

    false discovery rate (FDR) - NOT IMPLEMENTED
    FDR = FP / (FP + TN)
    false omission rate (FOR) - NOT IMPLEMENTED
    FOR = FN / (FN + TN)
    Threat score (TS) or Critical Success Index (CSI) - NOT IMPLEMENTED
    TS = TP / (TP + FN + FP)

=============================================================================
    CONFUSION MATRIX
    Answers the question: "is this record belongs to category X? yes or no?"
    an Upvote would be True Positive
    (FP: a Downvote would be False Positive -> thisis not taken into account here as Tamr does not take benefit of a Downvote)
    FP : a reclassification from category X to category Y would be a False Positive
    TP : a suggested classification that is validated is True Positive
    FN : a reclassification from category Y to category X would be a False Negative
    TN : a record that is not manually nor suggested in the category X would be True Negatives¶

=============================================================================
"""



import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

##################################
##################returns count of True Positives ################
def getTruePositiveForCategory(a_cat, a_df):
    df_work = a_df.loc[(a_df['manualClassificationId'] == a_cat) & (a_df['suggestedClassificationId'] == a_cat )]
    return df_work.shape[0]


#######################################
##################returns count of False Positives ################
def getFalsePositiveForCategory(a_cat, a_df):
    df_work = a_df.loc[(a_df['manualClassificationId'] != a_cat) & (a_df['suggestedClassificationId'] == a_cat )]
    return df_work.shape[0]

#######################################
##################returns count of False Negatives ################
def getFalseNegativeForCategory(a_cat, a_df):
    df_work = a_df.loc[(a_df['manualClassificationId'] == a_cat) & (a_df['suggestedClassificationId'] != a_cat )]
    return df_work.shape[0]

#######################################
##################returns count of True Negatives ################
def getTrueNegativeForCategory(a_cat, a_df):
    df_work = a_df.loc[(a_df['manualClassificationId'] != a_cat) &  (a_df['suggestedClassificationId'] != a_cat )]
    return df_work.shape[0]



#######################################
def f_buildMatrixAndConfidence(a_cat, a_dfDataForConfusionMatrix):
    cat_dict={}
    cat_dict['cat_path']=a_dfDataForConfusionMatrix.loc[a_dfDataForConfusionMatrix['manualClassificationId']==str(a_cat)]['manualClassificationPath'].iloc[0]
    cat_dict['cat_id'] = str(a_cat)
    cat_dict['true_positives'] = getTruePositiveForCategory(str(a_cat) ,a_dfDataForConfusionMatrix)
    cat_dict['true_negatives'] = getTrueNegativeForCategory(str(a_cat) ,a_dfDataForConfusionMatrix)
    cat_dict['false_positives'] = getFalsePositiveForCategory(str(a_cat) ,a_dfDataForConfusionMatrix)
    cat_dict['false_negatives'] = getFalseNegativeForCategory(str(a_cat) ,a_dfDataForConfusionMatrix)
    cat_dict['confidence'] = a_dfDataForConfusionMatrix.loc[a_dfDataForConfusionMatrix['manualClassificationId']==str(a_cat)]['suggestedClassificationConfidence'].mean()
    
    return cat_dict

#######################################
################## functions returns metric values ################
def f_precision_PPV (tp, fp):
    if (tp+fp) == 0:
        return np.nan
    else:
        return tp/(tp+fp)

#---------------------------------------
def f_NPV (tn, fn):
    if (tn+fn) == 0:
        return np.nan
    else:
        return tn/(tn+fn)

#---------------------------------------
def f_recall_TPR (tp, fn):
    if (tp+fn) == 0:
        return np.nan
    else:
        return tp/(tp+fn)

#--------------------------------------
def f_specificity_TNR (tp, fp):
    if (tp+fp) == 0:
        return np.nan
    else:
        return tp/(tp+fp)

#--------------------------------------
def f_miss_rate_FNR (tp, fn):
    if (tp+fn) == 0:
        return np.nan
    else:
        return fn/(tp+fn)

#---------------------------------------
def f_fall_out_FPR (tn, fp):
    if (tn+fp) == 0:
        return np.nan
    else:
        return fp/(tn+fp)


#---------------------------------------
def f_accuracy (tp, tn, fp, fn):
    if (tp+tn+fp+fn) == 0:
        return np.nan
    else:
        return (tp+tn)/(tp+tn+fp+fn)

                  
#---------------------------------------
#F1 score is the harmonic mean of precision and sensitivity
#---------------------------------------
def f_F1 (precision, recall):
    if (precision+recall) == 0:
        return np.nan
    else:
        return 2*(precision*recall)/(precision+recall)



######################################################################
if __name__ == "__main__":

    # Starting script
    print ('------ STARTING SCRIPT -----')
    print ('------ Loading config Yaml -----')
    with open('confusion_matrix_metrics_for_classification.yaml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)


    # read param
    # folder to write output - you can put input files too in it
    output_folder = config['output_folder']
    output_file = config['output_file']

    # classification_with_data file
    cfg_classification_with_data = config['classification_with_data_file']

    os.chdir(output_folder)


    #================== Calculation ===========================


    print ('------ Loading from input file -----')
    t1 = datetime.now()

    data = pd.read_csv(cfg_classification_with_data, encoding='utf-8', dtype=object)    
    data['suggestedClassificationConfidence'] = data['suggestedClassificationConfidence'].astype(float)
    t2 = datetime.now()
    print(f'Loading from input file - end - {t2 - t1} seconds.')
    print('')

    # subsetting where manualClassificationId not nan, which means we have an expertise validated answer given
    l_dfDataForConfusionMatrix = data.loc[~data['manualClassificationId'].isna()]

    # getting all categories that have been trained -  Question to solve: can it be `finalClassificationId` to use rather than `manualClassificationId` to consider? They should be the same, but can Tamr decide to choose another category than the one validated by Experts? 
    cat_list=l_dfDataForConfusionMatrix['manualClassificationId'].unique().tolist()

    # Building confusion matrix
    print ('------ Building confusion matrix -----')
    t1 = datetime.now()
    l_resultMatrix = pd.DataFrame(columns=['cat_id', 'cat_path', 'true_positives', 'true_negatives', 'false_positives', 'false_negatives', 'confidence'])
    l_resultMatrix = l_resultMatrix.append([f_buildMatrixAndConfidence(cat, l_dfDataForConfusionMatrix) for cat in cat_list], ignore_index=True)
    t2 = datetime.now()
    print(f'Building confusion matrix - end - {t2 - t1} seconds.')
    print('')

    # Calculating Metrics for each category
    print ('------ Calculating Metrics for each category -----')
    t1 = datetime.now()
    l_resultMatrix['precision_PPV'] = l_resultMatrix.apply(lambda x: f_precision_PPV(x['true_positives'], x['false_positives']), axis=1)
    l_resultMatrix['NPV'] = l_resultMatrix.apply(lambda x: f_NPV(x['true_negatives'], x['false_negatives']), axis=1)
    l_resultMatrix['recall_TPR'] = l_resultMatrix.apply(lambda x: f_recall_TPR(x['true_positives'], x['false_negatives']), axis=1)
    l_resultMatrix['fall_out_FPR'] = l_resultMatrix.apply(lambda x: f_fall_out_FPR(x['true_negatives'], x['false_positives']), axis=1)
    l_resultMatrix['specificity_TNR'] = l_resultMatrix.apply(lambda x: f_specificity_TNR(x['true_positives'], x['false_positives']), axis=1)
    l_resultMatrix['miss_rate_FNR'] = l_resultMatrix.apply(lambda x: f_miss_rate_FNR(x['true_positives'], x['false_negatives']), axis=1)
    l_resultMatrix['accuracy'] = l_resultMatrix.apply(lambda x: f_accuracy(x['true_positives'], x['true_negatives'],x['false_positives'] , x['false_negatives']), axis=1)
    l_resultMatrix['F1_score'] = l_resultMatrix.apply(lambda x: f_F1(x['precision_PPV'], x['recall_TPR']), axis=1)

    # Calculating Metrics overall
    print ('------ Calculating Metrics overall (adding a cat_id named global) -----')
    g_TP = l_resultMatrix['true_positives'].sum()
    g_TN = l_resultMatrix['true_negatives'].sum()
    g_FP = l_resultMatrix['false_positives'].sum()
    g_FN = l_resultMatrix['false_negatives'].sum()
    g_precision_PPV = f_precision_PPV(g_TP, g_FP)
    g_recall_TPR = f_recall_TPR(g_TP, g_FN)

    g_dict={}
    g_dict['cat_id'] = 'global'
    g_dict['cat_path'] = ''
    g_dict['true_positives'] = g_TP
    g_dict['true_negatives'] = g_TN
    g_dict['false_positives'] = g_FP
    g_dict['false_negatives'] =  g_FN
    g_dict['precision_PPV'] = g_precision_PPV
    g_dict['recall_TPR'] = g_recall_TPR
    g_dict['NPV'] = f_NPV(g_TN, g_FN)
    g_dict['specificity_TNR'] = f_specificity_TNR(g_TP, g_FP)
    g_dict['accuracy'] = f_accuracy(g_TP, g_TN, g_FP, g_FN)
    g_dict['F1_score'] = f_F1(g_precision_PPV, g_recall_TPR)

    ## Jerome 23rd Jan 2020 - removing confidence here as it represents average confidence of the confusion matrix, whcih does not bring value
#    g_dict['confidence'] = l_resultMatrix['confidence'].mean()
    g_dict['confidence'] = '---'

    l_resultMatrix = l_resultMatrix.append(g_dict, ignore_index=True)   
    t2 = datetime.now()
    print(f'Calculating Metrics for each category - end - {t2 - t1} seconds.')    
    # Save file
    print('')
    print ('------ Saving output file-----')    
    l_resultMatrix.to_csv(output_file, encoding='utf-8', index=None)
    print(f'Export file with {l_resultMatrix.shape[0]} rows (including global overall metrics)')
    print('')
    print('----- Ending Processing -----')
    print('----- ENDING SCRIPT -----')