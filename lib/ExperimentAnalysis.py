import numpy as np
import scipy

def getSummaryData(datList,dataShape):
    summaryData={}

    # solution data
    keyList=["solution_scorePrimal","solution_scoreDual","solution_scoreGap",\
            "solution_errorMargX","solution_errorMargY"]
    for k in keyList:
        summaryData[k]=[dat[k] for dat in datList]
        summaryData[k]=np.array(summaryData[k]).reshape(dataShape)[:,:,0]

    # relative pd gap
    kNew="solution_relativeGap"
    k1="solution_scoreGap"
    k2="solution_scorePrimal"
    keyList.append(kNew)
    summaryData[kNew]=summaryData[k1]/summaryData[k2]

    # partial time data
    newKeyList=["time_iterate","time_refine","time_measureBalancing","time_measureTruncation"]
    for k in newKeyList:
        summaryData[k]=[dat[k] for dat in datList]
        summaryData[k]=np.array(summaryData[k]).reshape(dataShape)

    keyList.extend(newKeyList)

    # global time data
    k="time_global"
    keyList.append(k)
    for dat in datList:
        summaryData[k]=[dat["timeList_global"][-1][-1] for dat in datList]
        summaryData[k]=np.array(summaryData[k]).reshape(dataShape)

    # global time data (time for last layer)
    k="time_global_penultimate_layer"
    keyList.append(k)
    summaryData[k]=[]

    for dat in datList:
        # first determine finest hierarchy level
        levelMax=np.max([x[0] for x in dat["timeList_global"]])
        # now gather all sparsity entry counts on penultimate level
        timeList=[]
        for x in dat["timeList_global"]:
            if x[0]==levelMax-1:
                timeList.append(x[-1])
        summaryData[k].append(timeList[-1])

    summaryData[k]=np.array(summaryData[k]).reshape(dataShape)
    # compute time on last layer
    k2="time_global_final_layer"
    summaryData[k2]=summaryData["time_global"]-summaryData[k]
    summaryData[k2]=np.array(summaryData[k2]).reshape(dataShape)
    keyList.append(k2)

    # sparsity data
    kMax="sparsity_max"
    kFinal="sparsity_final"
    keyList.extend([kMax,kFinal])
    summaryData[kMax]=[]
    summaryData[kFinal]=[]

    for dat in datList:
        # first determine finest hierarchy level
        levelMax=np.max([x[0] for x in dat["sparsity_muYAtomicEntries"]])
        # now gather all sparsity entry counts on finest level
        varNrList=[]
        for x in dat["sparsity_muYAtomicEntries"]:
            if x[0]==levelMax:
                varNrList.append(x[-1])
        summaryData[kMax].append(np.max(varNrList))
        summaryData[kFinal].append(varNrList[-1])

    for k in [kMax,kFinal]:
        summaryData[k]=np.array(summaryData[k]).reshape(dataShape)[:,:,0]

    return summaryData,keyList

def getSummaryDataSimple(datList,dataShape,keyList):
    summaryData={}

    for k in keyList:
        summaryData[k]=[dat[k] for dat in datList]
        summaryData[k]=np.array(summaryData[k]).reshape(dataShape)



    return summaryData

def getSummaryMeanStdDev(summaryData,keyList,numExperimentsKeep=None):
    # compute mean and variance of all keys
    summaryDataMean={}
    summaryDataStdDev={}
    for k in keyList:
        summaryDataMean[k]=np.mean(summaryData[k][:,:numExperimentsKeep],axis=1)
        summaryDataStdDev[k]=np.var(summaryData[k][:,:numExperimentsKeep],axis=1)**0.5

    return summaryDataMean,summaryDataStdDev
