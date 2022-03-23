import numpy as np
from .ScriptTools import ParameterType as ptype

def getDefaultParams():

    params={}
    # set some default values

    params["domdec_cellsize"]=4
    params["domdec_YThresh"]=1E-14

    params["sinkhorn_subsolver"]="LogSinkhorn"
    params["sinkhorn_error"]=1.E-6
    params["sinkhorn_error_rel"]=True

    params["parallel_iteration"]=True
    params["parallel_truncation"]=False
    params["parallel_balancing"]=True

    params["MPI_chunksize"]=10
    params["MPI_probetime"]=1E-3

    
    return params
# list of parameters to extract from command line
paramsListCommandLine=[\
        ["setup_tag",ptype.string],\
        ["setup_nIterations",ptype.integer]\
        ]

# list of parameters to extract from cfg file
paramsListCFGFile={\
        "setup_fn" : ptype.string,\
        #
        "domdec_cellsize" : ptype.integer,\
        "domdec_YThresh" : ptype.real,\
        #
        "sinkhorn_subsolver" : ptype.string,\
        "sinkhorn_error" : ptype.real,\
        "sinkhorn_error_rel" : ptype.boolean,\
        #
        "sinkhorn_eps" : ptype.real,\
        #
        "parallel_iteration" : ptype.boolean,\
        "parallel_truncation" : ptype.boolean,\
        "parallel_balancing" : ptype.boolean,\
        #
        "MPI_chunksize" : ptype.integer,\
        "MPI_probetime" : ptype.real,\
        #        
        }


