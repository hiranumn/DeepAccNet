<img src="figures/ipdlogo.png">

# DeepAccNet.py
Python-PyTorch implemenation of DeepAccNet described in https://www.biorxiv.org/content/10.1101/2020.07.17.209643v1

This method will estimate how good your protein models are using a metric called l-DDT (local distance difference test).

```
usage: DeepAccNet.py [-h] [--modelpath MODELPATH] [--pdb] [--csv] [--leaveTempFile] [--process PROCESS] [--featurize]
                     [--reprocess] [--verbose] [--bert] [--ensemble]
                     input ...

Error predictor network

positional arguments:
  input                 path to input folder or input pdb file
  output                path to output (folder path, npz, or csv)

optional arguments:
  -h, --help            show this help message and exit
  --modelpath MODELPATH, -modelpath MODELPATH
                        modelpath (Default: NatComm_standard)
  --pdb, -pdb           Running on a single pdb file instead of a folder (Default: False)
  --csv, -csv           Writing results to a csv file (Default: False)
  --leaveTempFile, -lt  Leaving temporary files (Default: False)
  --process PROCESS, -p PROCESS
                        Specifying # of cpus to use for featurization (Default: 1)
  --featurize, -f       Running only the featurization part (Default: False)
  --reprocess, -r       Reprocessing all feature files (Default: False)
  --verbose, -v         Activating verbose flag (Default: False)
  --bert, -bert         Run with bert features. Use extractBert.py to generate them. (Default: False)
  --ensemble, -e        Running with ensembling of 4 models. This adds 4x computational time with some overheads
                        (Default: False)

v0.0.1
```

- For the previous TensorFlow implementation, please see [here](https://github.com/hiranumn/DeepAccNet-TF).
- For the MSA version of DeepAccNet, please see [here](https://github.com/hiranumn/DeepAccNet-MSA).
- For the refinement script, please see the [modeling](modeling) folder.
- The dataset used to train this model can be accessed through [here](https://files.ipd.uw.edu/pub/DeepAccNet/decoys8000k.zip). Training splits can be accessed through [data](data)

# Softwares
- Python > 3.5
- PyTorch 1.3
- PyRosetta for DeepAccNet and DeepAccNet-Bert.
- [ProtTrans](https://github.com/agemagician/ProtTrans) and the ProtBert model (second one in the model availability table) for DeepAccNet-Bert.
- Tested on Ubuntu 20.04 LTS

# Example usages

Running on a folder of pdbs (foldername: ```samples```)
```
python DeepAccNet.py -r -v samples outputs
```
(For IPD users, please use the ```tensorflow``` conda environment)

# How to look at outputs
Output of the network is written to ```[input_file_name].npz```, unless you had the ```--csv``` flag on.
You can extract the predictions as follows.

```
import numpy as np

x = np.load("testoutput.npz")

lddt = x["lddt"]           # per residue lddt
estogram = x["estogram"]   # per pairwise distance e-stogram
mask = x["mask"]           # mask predicting native < 15
```
Perhaps ```lddt``` is the easiest place to start as it is per-residue quality score. You can simply take an average if you want a global score per protein structure. 

If you want to do something more involved, [check.ipynb](ipynbs/check.ipynb) is a good place to start.

# Trouble shooting
- If DeepAccNet.py returns an OOM (out of memory) error, your protein is probably too big. Try getting on titan instead of rtx2080 or run without gpu if running time is not your problem. You can also run it on cpus although it would be slow.
- If you get an import error for pyErrorPred, you probably moved the script out of the DeepAccNet folder. In that case, you would have to add pyErrorPred to python path or do so within the script. 
- Send an e-mail at hiranumn at cs dot washington dot edu.

# Updates
- Repo initialized 2020.7.20
- Transitioned to PyTorch 2020.11.3
