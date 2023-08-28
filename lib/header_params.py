import numpy as np
from .ScriptTools import ParameterType as ptype

def getDefaultParams():

    params={}
    # set some default values
    params["setup_dim"]=2

    params["hierarchy_top"]=4

    params["domdec_cellsize"]=4
    params["domdec_YThresh"]=1E-14
    params["domdec_refineAlpha"]=True

    params["sinkhorn_subsolver"]="SparseSinkhorn"
    params["sinkhorn_error"]=1.E-4
    params["sinkhorn_error_rel"]=False
    params["sinkhorn_max_iter"]=10000
    params["sinkhorn_inner_iter"]=10

    params["eps_schedule"]="default"
    params["eps_base"]=0.5
    params["eps_layerFactor"]=4.
    params["eps_layerSteps"]=2
    params["eps_stepsFinal"]=1
    params["eps_nIterations"]=1
    params["eps_nIterationsLayerInit"]=2
    params["eps_nIterationsGlobalInit"]=4
    params["eps_nIterationsFinal"]=1

    params["parallel_iteration"]=True
    params["parallel_truncation"]=False
    params["parallel_balancing"]=False
    params["parallel_refinement"]=False

    params["MPI_chunksize"]=10
    params["MPI_probetime"]=1E-3


    params["aux_printLayerConsistency"]=False
    
    
    params["aux_dump_finest"]=False
    params["aux_dump_finest_pre"]=False
    params["aux_dump_misc"]=False
    params["aux_dump_after_each_layer"]=False
    params["aux_dump_after_each_eps"]=False
    params["aux_dump_after_each_iter"]=False
    
    params["aux_evaluate_scores"]=True

    params["comparison_sinkhorn_truncation_thresh"]=1E-10
    params["comparison_verbose"]=False
    params["comparison_final_layer_manual"]=False
    params["comparison_sinkhorn_error"]=1E-6
    # Unable hybrid mode by default
    params["hybrid_mode"] = False
    params["unbalanced_mode"] = "balanced"
    
    return params
# list of parameters to extract from command line
paramsListCommandLine=[\
        ["setup_tag",ptype.string]\
        ]

# list of parameters to extract from cfg file
paramsListCFGFile={\
        "setup_fn1" : ptype.string,\
        "setup_fn2" : ptype.string,\
        "setup_dim" : ptype.integer,\
        #
        "hierarchy_depth" : ptype.integer,\
        "hierarchy_top" : ptype.integer,\
        #
        "domdec_cellsize" : ptype.integer,\
        "domdec_YThresh" : ptype.real,\
        "domdec_refineAlpha" : ptype.boolean,\
        #
        "sinkhorn_subsolver" : ptype.string,\
        "sinkhorn_error" : ptype.real,\
        "sinkhorn_error_rel" : ptype.boolean,\
        #
        "eps_schedule" : ptype.string,\
        "eps_base" : ptype.real,\
        "eps_layerFactor" : ptype.real,\
        "eps_layerSteps" : ptype.integer,\
        "eps_stepsFinal" : ptype.integer,\
        "eps_nIterations" : ptype.integer,\
        "eps_nIterationsLayerInit" : ptype.integer,\
        "eps_nIterationsGlobalInit" : ptype.integer,\
        "eps_nIterationsFinal" : ptype.integer,\
        #
        "parallel_iteration" : ptype.boolean,\
        "parallel_truncation" : ptype.boolean,\
        "parallel_balancing" : ptype.boolean,\
        "parallel_refinement" : ptype.boolean,\
        #
        "MPI_chunksize" : ptype.integer,\
        "MPI_probetime" : ptype.real,\
        #
        "aux_printLayerConsistency" : ptype.boolean,\
        "aux_dump_finest" : ptype.boolean,\
        "aux_dump_finest_pre" : ptype.boolean,\
        "aux_dump_misc" : ptype.boolean,\
        "aux_dump_after_each_layer" : ptype.boolean,\
        "aux_dump_after_each_eps" : ptype.boolean,\
        "aux_dump_after_each_iter" : ptype.boolean,\
        "aux_evaluate_scores" : ptype.boolean,\
        
        "comparison_sinkhorn_truncation_thresh" : ptype.real,\
        "comparison_verbose" : ptype.boolean,\
        "comparison_final_layer_manual" : ptype.boolean,\
        "comparison_sinkhorn_error" : ptype.real\
        
        
        }


