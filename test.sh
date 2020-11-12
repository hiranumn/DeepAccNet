#python DeepAccNet.py -r --verbose samples test/outputs1
#python DeepAccNet.py -r -e --verbose samples test/outputs2
#python extractBert.py samples test/outputs3
#python DeepAccNet.py -r --bert --verbose samples test/outputs3
#python extractBert.py samples test/outputs4
#python DeepAccNet.py -r --bert -e --verbose samples test/outputs4
#python DeepAccNet.py -r --verbose --csv samples test/outputs1.csv
#python extractBert.py samples samples
#python DeepAccNet.py -r --bert --verbose --csv samples test/outputs2.csv
#python DeepAccNet.py -r --verbose --pdb samples/tag0137.relaxed.al.pdb test/output1.npz
python DeepAccNet-noPyRosetta.py -r --verbose samples test/outputs5
python DeepAccNet-noPyRosetta.py -r -e --verbose samples test/outputs6
python extractBert.py samples test/outputs7
python DeepAccNet-noPyRosetta.py -r --bert --verbose samples test/outputs7
python extractBert.py samples test/outputs8
python DeepAccNet-noPyRosetta.py -r --bert -e --verbose samples test/outputs8
