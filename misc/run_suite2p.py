import os
from pathlib import Path
import shutil
import argparse
from tifffile import TiffFile
from natsort import natsorted
from merge_sync_mat import merge_mat_files
import yaml
import suite2p

parser = argparse.ArgumentParser(
                    prog='run_suite2P',
                    usage='run_suite2P data_path_1 ... data_path_n  -s [optional] save_path_1 ... save_path_n -m [optional]',
                    description='The program will run the basic suite2p pipeline for '
                                'all the recording files (tiff by default) found in '
                                'each of the specified data_path.' 
                                'if the -s (save_paths) option is not specified, ' 
                                'the program will save the suite2p output in data_path. ' 
                                'Otherwise, you can use the -s option for specyfing the '
                                'locations where to save the output, one for each of the '
                                'data_paths that are passed.')
                   
parser.add_argument('data_paths', nargs='+', help='(list [str], input data paths)')    
parser.add_argument('-s', '--save_paths', nargs='+', help='(list [str], output save paths)')                    
parser.add_argument('-m', '--merge', action='store_true', help='merge all the recordings found in the same data_path')

def runs2p(dataDirs, saveDirs, merge=False):

    if len(dataDirs) != len(saveDirs):
        print()

        print("> ERROR: The number of save paths should match the number of data paths!")
        return 0

    # set parameters to default
    ops = suite2p.default_ops()

    #load parameters specified in a s2p_ops.yaml, if present
    try:
        with open('s2p_ops.yaml', "r") as f:

            params = yaml.load(f, Loader=yaml.Loader)
            print('> using parameters from s2p_ops.yaml')

        for param in params:
            ops[param]= params[param]
    except:
        print('> No s2p_ops.yaml file found in current directory, all parameters set to default')

    for i,(dataDir,saveDir) in enumerate(zip(dataDirs,saveDirs)):

        print('\n> DATA PATH: {}'.format(dataDir))

        # check if multiple recordings exist in the current data_path
        files =  natsorted([file for file in os.listdir(dataDir) 
                    if file.endswith(ops['input_format'])])

        rec_names = ['_'.join(Path(file).stem.split('_')[:3]) for file in files]
        print('> Rec Files:')
        for rname in rec_names: print(rname)

        if merge:

            # check if there are .mat files to merge
            mat_names = natsorted([name+'.mat' for name in rec_names if name+'.mat' in os.listdir(dataDir)])

            if len(mat_names) == len(rec_names):

                print('> found .mat sync files. Merging:')
                for mname in mat_names: print(mname)

                # join the mat files
                offset = 0
                offsets = [offset]
                for file in files[:-1]:

                    with TiffFile(os.path.join(dataDir,file)) as tif:
                        # Get the number of pages, which corresponds to the number of frames
                        n_frames = len(tif.pages)
                        offset += n_frames
                        offsets.append(offset)
                
                # merge mat files
                merge_mat_files([os.path.join(dataDir,name) for name in mat_names],offsets)

            # run suite2p for all the recordings found
            print("\n> Running Suite2p ...")

            db = {
                'data_path': [dataDir],
                'save_path0': saveDir,
                    }
            suite2p.run_s2p(ops=ops, db=db)

        elif len(rec_names) > 1:

            # separate recording files in different directories

            for name in rec_names:

                new_dataDir = os.path.join(dataDir,name)
                
                os.mkdir(new_dataDir)

                if dataDir != saveDir:

                    new_saveDir = os.path.join(saveDir,name)
                    os.makedirs(new_saveDir)

                else:
                    new_saveDir = new_dataDir

                for file in files:

                    if name in file:

                        shutil.move(os.path.join(dataDir,file),new_dataDir)

                # run suite2p for all the recordings found
                print("\n> Running Suite2p ...")

                db = {
                    'data_path': [new_dataDir],
                    'save_path0': new_saveDir,
                        }
                suite2p.run_s2p(ops=ops, db=db)

        else:

            # run suite2p for all the recordings found
            print("\n> Running Suite2p ...")

            db = {
                'data_path': [dataDir],
                'save_path0': saveDir,
                    }
            suite2p.run_s2p(ops=ops, db=db)

if __name__ == '__main__':
    args = parser.parse_args()

    dataDirs =  args.data_paths
    if args.save_paths:
        saveDirs = args.save_paths
    else:
        saveDirs = dataDirs

    merge = args.merge

    runs2p(dataDirs,saveDirs,merge)