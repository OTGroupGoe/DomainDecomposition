import time
from mpi4py import MPI

"""
A simple implementation of parallel map for MPI.
The root job sends sub jobs to workers.
Only works on functions that are imported, declared beforehand on a global scope.
"""


# message codes
MSG_ROOT_all_done=0
MSG_ROOT_new_job=1
MSG_ROOT_set_func=3
MSG_ROOT_set_args_global=4
MSG_ROOT_new_job_list=5

MSG_WORKER_return_job=2

def sendProblem(comm,workerId, probId, data,multiProblem=False):
    """Master sends a problem to a worker.
    comm: MPI communication object
    workerId: rank of the worker
    probId: number of the problem in argument list
    data: arguments for the problem"""

    # send "new problem msg" first
    if multiProblem:
        comm.send(MSG_ROOT_new_job_list,workerId)
    else:
        comm.send(MSG_ROOT_new_job,workerId)
    # send problem id
    comm.send(probId,workerId)
    # send problem data
    comm.send(data,workerId)



def receiveSolution(comm):
    """Master listens to receiving a solution from one of the workers.
    comm: MPI communication object
    
    returns: (workerId,problemId,data)
    workerId: which worker finished a job
    problemId: what is the problem id of the job
    data: result data"""

    status=MPI.Status()
    msg=comm.recv(status=status)
    workerId=status.Get_source()
    problemId=comm.recv(source=workerId)
    data=comm.recv(source=workerId)
    
    return (workerId,problemId,data)
    
def receiveProblem(comm):
    """Worker receives problem from master (after receiving corresponding msg)"""
    probId=comm.recv(source=0)
    data=comm.recv(source=0)
    return (probId,data)


def sendSolution(comm,probId,data):
    """Worker sends solution of finished job back to master."""
    comm.send(MSG_WORKER_return_job, 0)
    comm.send(probId, 0)
    comm.send(data, 0)



def ParallelMap(comm,func,argList,argsGlobal=None,chunksize=1,probetime=None,\
        callableArgList=False,callableArgListLen=None,callableReturn=None):
    """Parallel map implementation with MPI.
    
    comm: MPI communication object
    func: function to be applied to each sample (must be known to workers)
    argList: list of data to which func is to be applied. Each entry of argList is a list of arguments that is to be plugged into f (see below).
    argsGlobal: optional list of global arguments that need to be used in each iteration.
    
    chunksize: how many problems are sent to a worker in one chunk, to let them work for longer periods independently
    probetime: if not None, the master sleeps for probetime between checking if a worker has reported back. this reduces the load of the worker process
    
    callableArgList: if True, argList is not actually a list, but a function that returns the args. Can be used to avoid having to set up all args in advance.
    callableArgListLen: length of abstract callable arg list
    callableReturn: if not None, for each returned job the master process will call callableReturn(jobId,jobResult).
        Can be used to avoid having to store all results before final post-processing.
    
    This returns:
    [f(*argsGlobal,*data) for data in argList]
    but the result is computed in parallel on the workers."""
    
    # number of available workers (number of processes-1)
    nWorkers=comm.Get_size()-1
    # number of total jobs to do (length of argList)
    if not callableArgList:
        nJobs=len(argList)
    else:
        nJobs=callableArgListLen
    # number of jobs that has been sent and has already been completed
    nJobsSent=0
    nJobsDone=0

    if callableReturn is None:
        # empty list to store results in
        result=[None for i in range(nJobs)]

    # initialize workers

    # list of free workers, initialize with list of all workers
    freeWorkerList=[i+1 for i in range(nWorkers)]

    for n in range(nWorkers):
        # send function to each worker    
        comm.send(MSG_ROOT_set_func,n+1)
        comm.send(func,n+1)

        # set global arguments to each worker
        comm.send(MSG_ROOT_set_args_global,n+1)
        if argsGlobal is not None:
            # send 1 if there are global arguments, followed by the global arguments
            comm.send(1,n+1)
            comm.send(argsGlobal,n+1)
        else:
            # only send 0 else
            comm.send(0,n+1)
        

    # main master loop
    contLoop=True
    while contLoop:
        if (nJobsSent<nJobs) and (len(freeWorkerList)>0):
            # while there are unsent jobs and free workers
            # pick free worker and next unsent job from list        
            curWorker=freeWorkerList.pop()
            curJob=nJobsSent
            if chunksize==1:
                # send single job to worker
                if not callableArgList:
                    sendProblem(comm,curWorker,curJob,argList[curJob])
                else:
                    sendProblem(comm,curWorker,curJob,argList(curJob))                    
                nJobsSent+=1
            else:
                # send multiple jobs to worker
                curChunkSize=min(chunksize,nJobs-nJobsSent)
                if not callableArgList:
                    sendProblem(comm,curWorker,curJob,argList[curJob:curJob+curChunkSize],multiProblem=True)
                else:
                    probData=[argList(i) for i in range(curJob,curJob+curChunkSize)]
                    sendProblem(comm,curWorker,curJob,probData,multiProblem=True)
                nJobsSent+=curChunkSize
            
        elif nJobsDone<nJobs:
            # else: all workers busy, but maybe not all jobs done
            
            if probetime is not None:
                while not comm.Iprobe(source=MPI.ANY_SOURCE):
                    time.sleep(probetime)

            # listen to next response from a worker that just finished a job
            curWorker,curJob,data=receiveSolution(comm)
            
            # write result into result list (or call callable return)
            if chunksize==1:
                if callableReturn is None:            
                    result[curJob]=data
                else:
                    callableReturn(curJob,data)
                nJobsDone+=1
            else:
                if callableReturn is None:            
                    result[curJob:curJob+len(data)]=data
                else:
                    for i,dat in enumerate(data):
                        callableReturn(curJob+i,dat)
                nJobsDone+=len(data)
                    
            # add worker back to free worker list
            freeWorkerList.append(curWorker)

            # when all jobs have been received, terminate main loop
            if nJobsDone==nJobs:
                contLoop=False
        
    
    
    if callableReturn is None:
        return result

def Close(comm):
    """Before the master process terminates it must send the "done"-signal to all workers for the programm to terminate successfully."""
    nWorkers=comm.Get_size()-1
    for n in range(nWorkers):
        comm.send(MSG_ROOT_all_done,n+1)

def Worker(comm):
    """Worker main routine. A worker process remains in this routine throughout the programm execution.
    When the main thread is finished it sends a "done"-signal to the worker that terminates the loop."""
    contLoop=True
    nArgsInGlobal=0
    
    while contLoop:
        # wait for a msg from the main process
        msg=comm.recv(source=0)
        
        # then parse the msg code:
        if msg==MSG_ROOT_all_done:
            # done: then terminate main loop
            contLoop=False
        elif msg==MSG_ROOT_new_job:
            # new job: parse the job data, apply function to data and return result
            probId,data=receiveProblem(comm)
            if nArgsInGlobal==0:
                # if no global arguments used
                sol=funcLocal(*data)
            else:
                # if global arguments used
                sol=funcLocal(*dataGlobal,*data)
            sendSolution(comm,probId,sol)
        elif msg==MSG_ROOT_new_job_list:
            # multiple new jobs
            probId,data=receiveProblem(comm)
            if nArgsInGlobal==0:
                # if no global arguments used
                sol=[funcLocal(*dat) for dat in data]
            else:
                # if global arguments used
                sol=[funcLocal(*dataGlobal,*dat) for dat in data]
            sendSolution(comm,probId,sol)            
        elif msg==MSG_ROOT_set_func:
            # set function on which to apply data
            funcLocal=comm.recv(source=0)
        elif msg==MSG_ROOT_set_args_global:
            # configure global arguments
            nArgsInGlobal=comm.recv(source=0)
            if nArgsInGlobal>0:
                dataGlobal=comm.recv(source=0)
            else:
                dataGlobal=[]
