"""
=============================================================================
Filename:       snapshot_classification_metrics.py
Author :        Jerome Gransac
Created:        2019-08-12
Last modified:  2019-08-12
Modified by:
Description:    Gather data from Tamr's output files and build a dataset with details on Match Prob, confidence and similarities for each pair of records in a cluster
                As well, calculates the MatchProb, confidence for the cluster
How to use:     Edit the variables in the file clusters_confidence_matchProb_prediction_similarities_results_config

				python clusters_confidence_matchprob_prediction_similarities_results.py

=============================================================================
"""



import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

##################################
##################returns a dict that is not used in the prcess - but exported ################
def getPairEntitiesByClustersWithRecords(a_ClustersWithMoreThan1Record):
    # clusters with more than one row : there are pairs predictions
    # building a dict: {key:persistentid , value: list of entityid}
    return (a_ClustersWithMoreThan1Record.groupby('persistentId')['entityId'].apply(list)).to_dict()


#######################################
def getListSignalFieldName(a_fieldListArg):
    l_list=[]
    for field in a_fieldListArg:
        l_list.append(field+'_signal')
    return l_list


#######################################
def getPredictionCalculation(a_clustersEntities, a_dfPredictions):
    predictions_reduced = a_dfPredictions[['entityId1', 'entityId2', 'prediction', 'confidence', 'matchProb']]

    # Match clusters with pairs on entity1
    df_predictions_entityid1_side = pd.merge(predictions_reduced, clusters_reduced, how='left', left_on='entityId1',
                                             right_on='entityId')

    df_predictions_entityid1_side.dropna(subset=['persistentId'], inplace=True)

    # rename columns
    df_predictions_entityid1_side.columns = ['entityId1', 'entityId2', 'prediction', 'confidence', 'matchProb',
                                             'entityId', 'Entity1PersistentId']
    df_predictions_entityid1_side = df_predictions_entityid1_side[
        ['entityId1', 'entityId2', 'prediction', 'confidence', 'matchProb',
         'Entity1PersistentId']]


    # Join above with clusters  on entity2. Then we get two clustersID : one related to entityid1, other to entityid2
    df_predictions_entityid2_side = pd.merge(df_predictions_entityid1_side, clusters_reduced, how='left',
                                             left_on='entityId2', right_on='entityId')

    # rename columns
    df_predictions_entityid2_side.dropna(subset=['persistentId'], inplace=True)
    df_predictions_entityid2_side.columns = ['entityId1', 'entityId2', 'prediction', 'confidence', 'matchProb',
                                             'Entity1PersistentId', 'entityId', 'Entity2PersistentId']

    # Then the pairs we want are the ones that have A clusterID for entity1 == clusterId for entity2
    df_PairsPredictionsByCluster = df_predictions_entityid2_side[
        df_predictions_entityid2_side['Entity1PersistentId'] == df_predictions_entityid2_side['Entity2PersistentId']]
    df_PairsPredictionsByCluster = df_PairsPredictionsByCluster[
        ['entityId1', 'entityId2', 'prediction', 'confidence', 'matchProb', 'Entity1PersistentId']]
    df_PairsPredictionsByCluster.columns = ['entityId1', 'entityId2', 'pair_prediction', 'pair_confidence',
                                            'pair_matchProb', 'persistentId']


    # Calculate mean of confidence, prediction and matchProb for each cluster (partition-like)
    df_PairsPredictionsByCluster['cluster_confidence'] = df_PairsPredictionsByCluster.groupby(
        'persistentId').pair_confidence.transform(np.mean)
    df_PairsPredictionsByCluster['cluster_prediction'] = df_PairsPredictionsByCluster.groupby(
        'persistentId').pair_prediction.transform(np.mean)
    df_PairsPredictionsByCluster['cluster_matchProb'] = df_PairsPredictionsByCluster.groupby(
        'persistentId').pair_matchProb.transform(np.mean)

    return df_PairsPredictionsByCluster


#################################################################
def enrichClustersCalculationswithSignals(a_fieldsList, a_clustersCalculations, a_dfPredictions):
    # merge withpredictions on extracolumns that are similarities signals
    l_list = a_fieldsList.copy()
    l_list.append('entityId1')
    l_list.append('entityId2')
    predictions_limited = a_dfPredictions[l_list]
    df_result = pd.merge(a_clustersCalculations, predictions_limited, how='inner', right_on=['entityId1', 'entityId2'],
                              left_on=['entityId1', 'entityId2'], sort=False)
    return df_result



######################################################################
if __name__ == "__main__":

    with open('clusters_confidence_matchprob_prediction_similarities_results_config.yaml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)


    # read param
    # folder to write output - you can put input files too in it
    output_folder = config['output_folder']
    output_file = config['output_file']

    # dedup_signals_predictions file
    dedup_signals_predictions = config['dedup_signals_predictions_file']
    # dedup_signals_predictions file
    dedup_published_clusters = config['dedup_published_clusters_file']
    # similarity_fields_list
    similarity_fields_list = config['similarity_fields_list']
    #do add similarity signals
    doSimilarities = config['do_similarity']

    os.chdir(output_folder)

    # Subsetting the datasets
    clustersEntitiesDict={}
    print ('------Start Script -----')
    print ('------Providing analytics -----')
    t1 = datetime.now()

    predictions = pd.read_csv(dedup_signals_predictions, encoding='utf-8', dtype=object)
    clusters = pd.read_csv(dedup_published_clusters, encoding='utf-8', dtype=object)
    clusters_reduced = clusters[['entityId', 'persistentId']]
    predictions.matchProb = predictions.matchProb.astype('float')
    predictions.prediction = predictions.prediction.astype('float')
    predictions.confidence = predictions.confidence.astype('float')
    predictions_reduced = predictions[['entityId1', 'entityId2', 'prediction', 'confidence', 'matchProb']]
    predictions_reduced = predictions_reduced.loc[predictions_reduced.matchProb > 0]


    # Subsetting data in clusters of Singletons and Lore than 1 records
    # Analytics to provide
    print(f'all predictions: {predictions.shape[0]}')
    print(f'predictions with matchProb >0: {predictions_reduced.shape[0]}')
    print(f'nb of Entities: {clusters_reduced.shape[0]}')
    entityGroupedByCluster = clusters_reduced.groupby('persistentId', as_index=False).agg({"entityId": "count"})
    entityGroupedByCluster.rename(columns={
        "entityId": "nb_rec"}, inplace=True)

    clustersWithMoreThan1Record = entityGroupedByCluster.loc[entityGroupedByCluster.nb_rec > 1]
    clustersWith1Record = entityGroupedByCluster.loc[entityGroupedByCluster.nb_rec == 1]
    df_clustersOneRecords = clusters_reduced[ clusters_reduced['persistentId'].isin(clustersWith1Record['persistentId'].tolist())]
    df_ClustersEntitiesWithMoreThan1Pair = clusters_reduced[clusters_reduced['persistentId'].isin(clustersWithMoreThan1Record['persistentId'].tolist())]

    print(f'clusters With More Than 1 Record: {len(clustersWithMoreThan1Record)}')
    print(f'clusters With 1 Record: {len(clustersWith1Record)}')
    t2 = datetime.now()
    print(f'------END Analytics - end - {t2 - t1}------ ')

    # PROCESS
    # find which cluster(s) contains(s) entity 1 from pairs + find which cluster(s) contains(s) entity 2 from pairs
    # when ClusterID for entity1 == ClusterId for entity2, then the pair (entity1 , entity2) belongs to the Cluster
    # calculate prediction and confidence for the cluster based on the pairs stats
    # add extra columns signals
    # save

    print('')
    print ('------Start processing -----')
    print(f'Pair Entities By Clusters With Records - start')
    print('')
    #building a dataset file with dictionary key/value: {clusterId : [entities in cluster]}
    # no use for the script, just providing the result into a file
    t1 = datetime.now()
    clustersEntitiesDict = getPairEntitiesByClustersWithRecords(df_ClustersEntitiesWithMoreThan1Pair)
    fout = '1-clustersEntitiesDict.txt'
    fo = open(fout, 'w')
    for k, v in clustersEntitiesDict.items():
        fo.write(str(k) + ' >>> ' + str(v) + '\n\n')
    t2 = datetime.now()
    print(f'FYI: Produced an intermediary dictionary file named {fout}')
    print(f'Pair Entities By Clusters With Records - end - {t2 - t1} - Length: {len(clustersEntitiesDict)}')

    # determining pairs that belong to a cluster - make calculation cluster confidence and matchProb
    print('')
    print(f'Prediction Calculation - start')
    t1 = datetime.now()
    fout='dfClustersEntitiesCalculation.csv'
    dfClustersEntitiesPrediction = getPredictionCalculation(df_ClustersEntitiesWithMoreThan1Pair, predictions)
    dfClustersEntitiesPrediction.to_csv(fout, encoding='utf-8', index=None)
    t2 = datetime.now()
    print(f'Produced an intermediary file named {fout}')
    print(f'Prediction Calculation - end - {t2 - t1} - Shape: {dfClustersEntitiesPrediction.shape}')

    # adding similarities if True in config file
    # make sure the similarities field are mentioned
    if doSimilarities:

        l_listReadySignalFields = getListSignalFieldName(similarity_fields_list)
        if l_listReadySignalFields:
            print('')
            print(f'Field for Signals: = {l_listReadySignalFields}')
            print(f'Adding Signals similarities - start')
            t1 = datetime.now()
            dfResultSimilaritiesFinal = enrichClustersCalculationswithSignals(l_listReadySignalFields, dfClustersEntitiesPrediction, predictions)
            t2 = datetime.now()
            print(f'Clusters With Similarities Signals - end - {t2 - t1} - Shape: {dfResultSimilaritiesFinal.shape}')
        else:
            print('Similarities fields list is empty. Provide one list with ML fields as they are named in the unified dataset')



    # Save file
    dfResultSimilaritiesFinal.to_csv(output_file, encoding='utf-8', index=None)
    print('')
    print(f'Export file with {dfResultSimilaritiesFinal.shape[0]} rows (valid pairs prediction for given clusters.')
    print('----- End Processing -----')
    print('----- END SCRIPT -----')