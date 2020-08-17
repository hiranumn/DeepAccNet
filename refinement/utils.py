#!/usr/bin/python
import os,sys
import copy
import numpy  as np

SCRIPTPATH = os.environ.get("SCRIPTPATH")

def check_files(fs):    
    missing = [f for f in fs if not os.path.exists(f)]
    if missing != []:
        sys.exit("Missing file: %s, terminate!"%(','.join(missing)))
        
def aa3toaa1(threecode):
    aamap = {'ALA':'A','CYS':'C','CYD':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y','MSE':'M','CSO':'C','SEP':'S'}
    return aamap[threecode]

def pdb_in_resrange(pdb,newname,resrange,skip_alt=True,exres=False):
    cont = open(pdb)
    newpdb = open(newname,'w')
    newcont = []
    for line in cont:
        if line[:4] not in ['ATOM','HETA']:
            continue
        resno = int(line[22:26])
        alt = line[16]
        if skip_alt and (alt not in [' ','A']):
            continue
        if exres:
            if resno not in resrange:
                newcont.append(line)
        else:
            if resno in resrange:
                newcont.append(line)
    newpdb.writelines(newcont)

def pdb2res(pdbfile,bychain=False,chaindef=[],withchain=False,single=False):
    pdbcont = open(pdbfile)
    restype = {}
    for line in pdbcont:
        if line[:4]!='ATOM':
            continue

        if line[12:16].strip() == 'CA':
            res = int(line[22:26])
            chain = line[21]
            if withchain:
                res = '%s%04d'%(chain,res)

            if line[26] != ' ':
                continue
            if chaindef != [] and chain not in chaindef:
                continue

            char = line[17:20]
            if char in ['HIP','HID','HIE']:
                char = 'HIS'
            elif char == 'CSS':
                char = 'CYS'
            if single:
                char = threecode_to_alphabet(char)

            if bychain:
                if chain not in restype:
                    restype[chain] = {}
                restype[chain][res] = char
            else:
                restype[res] = char
    return restype

def pdb2crd(pdbfile,opt,as_numpy=True):
    pdbcont=open(pdbfile)
    crd={}
    for line in pdbcont:
        if line[:4]!='ATOM':
            continue
        resno = int(line[22:26])
        restype = line[16:20].strip()
        atmtype = line[12:16].strip()
        if opt == 'CA':
            if line[12:16] == ' CA ':
                v = [float(line[30+i*8:38+i*8]) for i in range(3)]
                if as_numpy: v = np.array(v)
                crd[resno] = v
        elif opt == 'CB':
            if (restype == 'GLY' and line[12:16] == ' CA ') or line[12:16] == ' CB ':
                v = [float(line[30+i*8:38+i*8]) for i in range(3)]
                if as_numpy: v = np.array(v)
                crd[resno] = v
        else:
            if resno not in crd:
                crd[resno] = {}
            v = [float(line[30+i*8:38+i*8]) for i in range(3)]
            if as_numpy: v = np.array(v)
            crd[resno][atmtype] = v
    pdbcont.close()
    return crd

def pdb2fa(pdb,outfa='',gap=True):
    if outfa == '':
        outfa = pdb.replace('pdb','fa')
    cont = open(pdb)
    savecont = ['>%s\n'%pdb[:-4]]
    char = ''
    prvres = -999
    for line in cont:
        if line[:4]!='ATOM':
            continue
        resno = int(line[22:26] )
        if resno == prvres:
            continue
        if resno-prvres > 1 and prvres != -999 and gap:
            char += '-'*(resno-prvres-1)
        seq = line[16:20].strip()
        char += aa3toaa1(seq)
        prvres = resno
    char+= '\n'
    savecont.append(char)
    savefile = open(outfa,'w')
    savefile.writelines(savecont)

def reset_score(silent,score,value):
    cont = []
    for l in open(silent):
        if l.startswith('SCORE'):
            words = l[:-1].split()
            if 'description' in l:
                # insert just before tag
                k_score = words.index(score)
                cont.append(l)
            else:
                tag = words[-1]
                part1 = words[1:k_score]
                part2 = words[k_score+1:]
                if isinstance(value,dict):
                    newl = 'SCORE:'+' %10s'*len(part1)%tuple(part1)+' %10s'%value[tag]+' %10s'*len(part2)%tuple(part2)+'\n'
                else:
                    newl = 'SCORE:'+' %10s'*len(part1)%tuple(part1)+' %10s'%value+' %10s'*len(part2)%tuple(part2)+'\n'
                cont.append(newl)
        else:
            cont.append(l)
    return cont

def all_add_score(silent,score,value):
    cont = []
    for l in open(silent):
        if l.startswith('SCORE'):
            words = l[:-1].split()
            if 'description' in l:
                # insert just before tag
                newl = l.replace('description','%10s description'%score)
                cont.append(newl)
            else:
                tag = words[-1]
                part1 = words[1:-1]
                part2 = [words[-1]]
                newl = 'SCORE:'+' %10s'*len(part1)%tuple(part1)+' %10d'%value+' %10s'*len(part2)%tuple(part2)+'\n'
                cont.append(newl)
        else:
            cont.append(l)
    return cont

def scparse(scfile,tags):
    ks = []
    header_printed = False
    outcont = []
    for line in open(scfile):
        if not line.startswith("SCORE:"):
            continue
        words = line[:-1].split()
        if 'description' in line:
            ks = []
            printstr = '%-30s'%'tag'
            for key in tags:
                if key in words:
                    ks.append(words.index(key))
                    printstr += ' %10s'%key
                elif key[0] == '[' and key[-1] == ']' and ('+' in key or '-' in key):
                    sig = 1.0
                    s = 1
                    ks.append([])
                    for i,a in enumerate(key[1:-1]):
                        if a == '+':
                            ks[-1].append( (sig,words.index(key[s:i+1])) )
                            sig = 1.0
                            s = i+2
                        elif a == '-':
                            ks[-1].append( (sig,words.index(key[s:i+1])) )
                            sig = -1.0
                            s = i+2
                    ks[-1].append( (sig,words.index(key[s:-1])) )
                
                else:
                    continue
            nheader = len(words)
            if not header_printed:
                outcont.append( printstr )
                header_printed = True
        else:
            try:
                printstr = '%-30s'%words[-1]
                for i,k in enumerate(ks):
                    if isinstance(k,list):
                        val = 0.0
                        for sig,kk in k:
                            val += sig*float(words[kk])
                        printstr += ' %10.3f'%val
                    else:
                        printstr += ' %10s'%words[k]
                outcont.append( printstr )
            except:
                continue
    return outcont

def read_d0mtrx(pdb,nres=-1,byresno=False):
    crds = pdb2crd(pdb,'CB')
    reslist = list(crds.keys())
    reslist.sort()
    if nres == -1: nres = len(reslist)

    d0mtrx = [[0.0 for i in range(nres)] for j in range(nres)]
    for i,ires in enumerate(reslist):
        for j,jres in enumerate(reslist):
            dxyz = crds[ires]-crds[jres]
            d = np.sqrt(np.dot(dxyz,dxyz))
            if byresno:
                d0mtrx[ires-1][jres-1] = d
                d0mtrx[jres-1][ires-1] = d
            else:
                d0mtrx[i][j] = d
                d0mtrx[j][i] = d
    return d0mtrx

def replace_file(reffile,outfile,charlist,replist):
    cont = copy.copy(open(reffile).readlines()) 
    for j,char in enumerate(charlist):
        for i,line in enumerate(cont):
            #print(char,line)
            if char in line:
                cont[i] = line.replace(char,replist[j])
    newfile = open(outfile,'w')
    newfile.writelines(cont)
    newfile.close()

def list2part(inlist):
    partlist = []
    for i,comp in enumerate(inlist):
        if isinstance(comp,int):
            if i == 0 or abs(comp-prv) != 1:
                partlist.append([comp])
            else:
                partlist[-1].append(comp)
        elif isinstance(comp,str):
            if i == 0 or comp != prv:
                partlist.append([comp])
            else:
                partlist[-1].append(comp)
        prv = comp
    return partlist

def trim_lessthan_3(ulrin,nres):
    regs = list2part(ulrin)
    for reg in copy.copy(regs):
        if len(reg) < 3:
            regs.remove(reg)
            
    if len(regs) == 0:
         return []
     
    ulrres = []
    for i,reg in enumerate(regs[:-1]):
        if reg[0] <= 3:
            reg = list(range(1,reg[-1]+1))
        if regs[i+1][0]-reg[-1] <= 3:
            reg += list(range(reg[-1]+1,regs[i+1][0]))
        ulrres += reg
     
    if nres-regs[-1][-1] <= 3:
        regs[-1] = range(regs[-1][0],nres+1)
    ulrres += regs[-1]
    return ulrres
    
def sandwich(inlist,super_val,infer_val):
    inlist_cp = copy.copy(inlist)
    for i,val in enumerate(inlist_cp[:-1]):
        if i == 0:
            continue
        if inlist_cp[i-1] == super_val and inlist_cp[i+1] == super_val:
            inlist[i] = super_val

    inlist_cp = copy.copy(inlist)
    for i,val in enumerate(inlist_cp[:-1]):
        if i == 0:
            continue
        if inlist_cp[i-1] == infer_val and inlist_cp[i+1] == infer_val:
            inlist[i] = infer_val
