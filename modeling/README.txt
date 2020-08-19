* Required libraries:
- pyRosetta3
- Rosetta
- gnu parallel
- Python libraries: [scipy, numpy]
- tensorflow setup for CPUs to run DeepAccNet (NOTE: DeepAccNet is called only through CPUs during refinement run)

* How to configure:
Please edit your setup.sh file to specify the paths where library installations exist.
Note that all DeepAccNet runs in CPUs in the refinement process -- please make sure tensorflow also works in CPU.

* Preprocessing/Required inputs:
- t000_.3mers & t000_.9mers : Rosetta fragment library files, for 3mer/9mer libraries, respectively
- init.pdb : Starting model as pdb format
- init.npz : DeepAccNet prediction on init.pdb

* How to run:
An example SLURM script "slurm_example.sh" is provided. Below is more detailed description:

1. Setup environment
   > source $SCRIPTPATH/setup.sh 

2. Prepare a directory containing [t000_.3mers, t000_.9mers, init.pdb, init.npz]
   Getting init.npz from init.pdb: > python $DANPATH/DeepAccNet.py --pdb init.pdb

3. Run initial model diversification at a directory 'idiv':
   > python $SCRIPTPATH/MainDiversification.py idiv init.npz (will take a few hours using 60 cores)

4. Run iterative intensification at a directory 'ihyb.a' (aggressive mode as default):
   > python $SCRIPTPATH/MainIteration.py idiv/pick.Q.out init.npz ihyb.a (will take a few hours using 60 cores)
   
4-1. Optionally, run a "conservaative-mode" iterative intensification separately at a directory 'ihyb.c'
   > python $SCRIPTPATH/MainIteration.py idiv/pick.Q.out init.npz ihyb.c -opt cR2D (will take a few hours using 60 cores)

5. Post-process the iteration results to get a representative model (takes 5~10 minutes in a single core)
   > python $SCRIPTPATH/PostProcess.py [ihyb.a/ihyb.c]
   The final model will be 'Qsel.avrg.relaxed.pdb'!

