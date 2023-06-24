import json
from os import listdir

files = listdir('ElonMusk/2023-02-08/retweeters/')
files = [file for file in files if '.json' in file]
retweeter_l = []
for file in files:
    with open('ElonMusk/2023-02-08/retweeters/' + file, 'r') as fp:
        obj = json.load(fp= fp)
        for k in obj.keys():
            retweeter_l.append(k)
print(len(retweeter_l))
