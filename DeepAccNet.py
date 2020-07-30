import sys
import argparse
import os
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
import multiprocessing

def main():
    #####################
    # Parsing arguments
    #####################
    parser = argparse.ArgumentParser(description="Error predictor network",
                                     epilog="v0.0.1")
    parser.add_argument("input",
                        action="store",
                        help="path to input folder or input pdb file")
    
    parser.add_argument("output",
                        action="store", nargs=argparse.REMAINDER,
                        help="path to output (folder path, npz, or csv)")
    
    parser.add_argument("--pdb",
                        "-pdb",
                        action="store_true",
                        default=False,
                        help="Running on a single pdb file instead of a folder (Default: False)")
    
    parser.add_argument("--csv",
                        "-csv",
                        action="store_true",
                        default=False,
                        help="Writing results to a csv file (Default: False)")
    
    parser.add_argument("--ensemble",
                        "-e", 
                        action="store_true",
                        default=False,
                        help="Running with ensembling of 4 models.  This adds 4x computational time with some overheads (Default: False)")
    
    parser.add_argument("--leaveTempFile",
                        "-lt",
                        action="store_true",
                        default=False,
                        help="Leaving temporary files (Default: False)")
    
    parser.add_argument("--verbose",
                        "-v",
                        action="store_true",
                        default=False,
                        help="Activating verbose flag (Default: False)")
    
    parser.add_argument("--process",
                        "-p", action="store",
                        type=int,
                        default=1,
                        help="Specifying # of cpus to use for featurization (Default: 1)")
    
    parser.add_argument("--gpu",
                        "-g", action="store",
                        type=int,
                        default=0,
                        help="Specifying gpu device to use (default gpu0)")
    
    parser.add_argument("--featurize",
                        "-f",
                        action="store_true",
                        default=False,
                        help="Running only the featurization part(Default: False)")
    
    parser.add_argument("--reprocess",
                        "-r", action="store_true",
                        default=False,
                        help="Reprocessing all feature files (Default: False)")
    
    args = parser.parse_args()
    
    ################################
    # Checking file availabilities #
    ################################
    csvfilename = "result.csv"
    
    # made outfolder an optional positinal argument. So check manually it's lenght and unpack the string
    if len(args.output)>1:
        print(f"Only one output folder can be specified, but got {args.output}", file=sys.stderr)
        return -1
    
    if len(args.output)==0:
        args.output = ""
    else:
        args.output = args.output[0]

    if args.input.endswith('.pdb'):
        args.pdb = True
    
    if args.output.endswith(".csv"):
        args.csv = True
        
    if not args.pdb:
        if not isdir(args.input):
            print("Input folder does not exist.", file=sys.stderr)
            return -1
        
        #default is input folder
        if args.output == "":
            args.output = args.input
        else:
            if not args.csv and not isdir(args.output):
                if args.verbose: print("Creating output folder:", args.output)
                os.mkdir(args.output)
            
            # if csv, do it in place.
            elif args.csv:
                csvfilename = args.output
                args.output = args.input
          
    else:
        if not isfile(args.input):
            print("Input file does not exist.", file=sys.stderr)
            return -1
        
        #default is output name with extension changed to npz
        if args.output == "":
            args.output = os.path.splitext(args.input)[0]+".npz"

        if not(".pdb" in args.input and ".npz" in args.output):
            print("Input needs to be in .pdb format, and output needs to be in .npz format.", file=sys.stderr)
            return -1
        
    script_dir = os.path.dirname(__file__)
    base = os.path.join(script_dir, "models/")
    modelpath = join(base, "regular")
        
    if args.ensemble:
        for i in range(1,5):
            if not isdir(modelpath+"_rep"+str(i)):
                print("Model checkpoint does not exist", file=sys.stderr)
                return -1
    else:        
        if not isdir(modelpath+"_rep1"):
            print("Model checkpoint does not exist", file=sys.stderr)
            return -1
        
    ##############################
    # Importing larger libraries #
    ##############################
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    script_dir = os.path.dirname(__file__)
    sys.path.insert(0, script_dir)
    import pyErrorPred
        
    num_process = 1
    if args.process > 1:
        num_process = args.process
        
        #########################
    # Getting samples names #
    #########################
    if not args.pdb:
        samples = [i[:-4] for i in os.listdir(args.input) if isfile(args.input+"/"+i) and i[-4:] == ".pdb" and i[0]!="."]
        ignored = [i[:-4] for i in os.listdir(args.input) if not(isfile(args.input+"/"+i) and i[-4:] == ".pdb" and i[0]!=".")]
        if args.verbose: 
            print("# samples:", len(samples))
            if len(ignored) > 0:
                print("# files ignored:", len(ignored))

        ##############################
        # Featurization happens here #
        ##############################
        inputs = [join(args.input, s)+".pdb" for s in samples]
        tmpoutputs = [join(args.output, s)+".features.npz" for s in samples]
        
        if not args.reprocess:
            arguments = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs)) if not isfile(tmpoutputs[i])]
            already_processed = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs)) if isfile(tmpoutputs[i])]
            if args.verbose: 
                print("Featurizing", len(arguments), "samples.", len(already_processed), "are already processed.")
        else:
            arguments = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs))]
            already_processed = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs)) if isfile(tmpoutputs[i])]
            if args.verbose: 
                print("Featurizing", len(arguments), "samples.", len(already_processed), "are re-processed.")

        if num_process == 1:
            for a in arguments:
                pyErrorPred.process(a)
        else:
            pool = multiprocessing.Pool(num_process)
            out = pool.map(pyErrorPred.process, arguments)
            
        # Exit if only featurization is needed
        if args.featurize:
            return 0
        
        if args.verbose: print("using", modelpath)
        
        ###########################
        # Prediction happens here #
        ###########################
        samples = [s for s in samples if isfile(join(args.output, s+".features.npz"))]
        result = pyErrorPred.predict(samples,
                                        modelpath,
                                        args.output,
                                        num_blocks=5,
                                        num_filters=128,
                                        verbose=args.verbose,
                                        ensemble=args.ensemble,
                                        csv = args.csv)
        
        if not args.csv:        
            if args.ensemble:
                pyErrorPred.merge(samples,
                                  args.output,
                                  verbose=args.verbose)

            if not args.leaveTempFile:
                pyErrorPred.clean(samples,
                                  args.output,
                                  verbose=args.verbose,
                                  multimodel=False,
                                  noEnsemble=not(args.ensemble))
        else:
            # Take average of outputs
            csvfile = open(csvfilename, "w")
            csvfile.write("sample\tcb-lddt\n")
            for s in samples:
                line = "%s\t%.4f\n"%(s, np.mean(result[s]))
                csvfile.write(line)
            csvfile.close()
            
            # Clean feature files
            pyErrorPred.clean(samples,
                              args.output,
                              verbose=args.verbose,
                              multimodel=False,
                              noEnsemble=True)
            
            
    # Processing for single sample
    else:
        infilepath = args.input
        outfilepath = args.output
        infolder = "/".join(infilepath.split("/")[:-1])
        insamplename = infilepath.split("/")[-1][:-4]
        outfolder = "/".join(outfilepath.split("/")[:-1])
        outsamplename = outfilepath.split("/")[-1][:-4]
        feature_file_name = join(outfolder, outsamplename+".features.npz")
        if args.verbose: 
            print("only working on a file:", outfolder, outsamplename)
        # Process if file does not exists or reprocess flag is set
        
        if (not isfile(feature_file_name)) or args.reprocess:
            pyErrorPred.process((join(infolder, insamplename+".pdb"),
                                feature_file_name,
                                args.verbose))   
        if isfile(feature_file_name):
            pyErrorPred.predict([outsamplename],
                    modelpath,
                    outfolder,
                    num_blocks=5,
                    num_filters=128,
                    verbose=args.verbose,
                    ensemble=args.ensemble)
            
            if args.ensemble:
                pyErrorPred.merge([outsamplename],
                                  outfolder,
                                  verbose=False)

            if not args.leaveTempFile:
                pyErrorPred.clean([outsamplename],
                                  outfolder,
                                  verbose=args.verbose,
                                  multimodel=False,
                                  noEnsemble=not(args.ensemble))
        else:
            print(f"Feature file does not exist: {feature_file_name}", file=sys.stderr)
            
            
if __name__== "__main__":
    main()