CONFIGS = {
# Initial diversification params
    'nstruct_div': 10, #per job
    'njobs_div'  : 100, # NSTRUCT*NJOBS = NTOTAL per option (aggr/cons)

## All below are for iterative process
#Basic run opt
    'niter'                  : 30,
    'debug'                  : False,
    'simple'                 : False,
    'nproc'                  : 60,           #Ncores to use
    'max_min_terminate'      : 240.0,        #kill if each iter last longer than this
    'npool'                  : 50,           #N members in the pool

# Scoring params
    'cstweight'              : 0.2,
    'cst_fa_weight'          : 1.0,
    #'cencst'                 : '',
    #'facst'                  : '',
    'pool_update'            : 'Q',   # [standard/Q/Qlocal/...]

# Sampling params
    'mulfactor_phase0'       : 1.0,
    'ntempl_per_job'         : 5,            #how may templates per each hybrid run
    'cross2_autocst'         : True,
    'reconstruct_every_iter' : True,
    'xml_cross1_suffix'      : 'cross.xml',  #use faster hybrid
    'xml_cross2_suffix'      : 'cross2.xml',
    'xml_mut_suffix'         : 'mut.xml',
    
# GA/CSA params
    'nseed'                  : 10,
    'nperseed'               : 4,            # 4-n for cross / n for mut
    'ngen_per_job'           : 3,            #total jobs: NSEED*NPERSEED*NGEN_PER_JOB : 120
    'nmutperseed'            : {0:2},        # key: phase value: nmut
    'dcut0'                  : 0.4,          #max dcut
    'dcut_min_scale'         : 0.5,          #DCUTMIN:DCUT0*DCUT_MIN_SCALE
    'n0_to_reset'            : 3,            #
    'simlimit_base'          : 0.05,
    'recomb'                 : "random",
    'recomb_iter'            : 5, #unused if <0
    
# Deformation factor params
    'penalize_wrt_init'      : 1,
    'iha'                    : 25.0,
    
# Extra options
    'e2cst_args'             : '-pcore 0.8 0.8 0.9',
    'native'                 : '',
    'symmetric'              :  0,
    'symmdef'                : ''
}
