import json
s = 0
for i in range(14):
    with open('ElonMusk/2023-02-14/retweeters/%d_retweeters.json' % i, 'r') as fp:
        obj = json.load(fp= fp)
        print('%d_retweeters.json || ' % i, end= ' ')
        print(len(obj.keys()))
        s += len(obj.keys())
print('Total || ', s)