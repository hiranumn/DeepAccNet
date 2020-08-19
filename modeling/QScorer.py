import os,sys
import numpy as np
from pyrosetta import *
from config import CONFIGS

DLPATH = os.environ['DANPATH']

class QScorer:
    def __init__(self):
        self.ncore = CONFIGS['nproc']

    def score(self,poses):
        scores = self.score_hack(poses)
        return scores
    
    def score_hack(self,poses,clean=True):
        os.system('mkdir tmp 2>/dev/null')
        for i,pose in enumerate(poses):
            pose.dump_pdb("tmp/tmp.%02d.pdb"%i)
        os.chdir('tmp')
        os.system('python %s/DeepAccNet.py -p %d ./ > DAN.logerr'%(DLPATH,self.ncore))

        n = len(poses)
        npzs = ['tmp.%02d.npz'%i for i in range(n)]
        scores = np.zeros(n)
        for i,npz in enumerate(npzs):
            scores[i] = np.mean(np.load(npz)['lddt'])
        os.chdir('..')
        if clean: os.system('rm -rf tmp')
        return scores

def add_score_to_outsilent(inf,outf,tags,scores):
    out = open(outf,'w')
    for l in open(inf):
        if l.startswith('SCORE'):
            words = l[:-1].split()
            if 'description' in l:
                newl = l[:l.index('description')]+'%-8s'%"Q"+'description\n'
            else:
                tag = words[-1]
                if tag not in tags:
                    print("Pass tag %s"%tag)
                    continue
                score = scores[tags.index(tag)]
                newl = l[:l.index(tag)]+'%-8.3f %s\n'%(score,tag)
            out.write(newl)
        else:
            out.write(l)
    out.close()
    
def main(insilent,outsilent,n=50):    
    init() #PyRosetta
    tags = []
    poses = []
    pis = rosetta.core.import_pose.pose_stream.SilentFilePoseInputStream(insilent)
    while pis.has_another_pose():
        ss = pis.next_struct()
        pose = Pose()
        ss.fill_pose(pose)
        poses.append(pose)
        tags.append(ss.decoy_tag())
        
    scorer = QScorer()
    scores = scorer.score(poses)
    idx = np.argsort(-scores)
    if len(idx) > n:
        tags = [tags[i] for i in idx[:n]]

    print(tags)
    add_score_to_outsilent(insilent,outsilent,tags,scores)

if __name__ == "__main__":
    insilent = sys.argv[1]
    if len(sys.argv) > 2:
        outsilent = sys.argv[2]
    else:
        outsilent = insilent.replace('.out','.Q.out')
    n = CONFIGS['npool']
        
    main(insilent,outsilent,n)
