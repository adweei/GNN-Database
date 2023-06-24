import json
import pandas as pd


user_prof = dict()
col = list()
col_name = ['id', 'location', 'followers_count', 'following_count', 'tweet_count', 'verified', 'created_at', 'protected']
for i in range(14):
    with open('ElonMusk/2023-02-16/user_profile/%d.json' % i, 'r') as fp:
        obj = json.load(fp= fp)
        for k in obj.keys():
            if obj[k]:
                x = obj[k]
                row = list()
                row.append(k)
                for name in col_name:
                    if name in x.keys():
                        row.append(x[name])
                col.append(row)
        #user_prof.update(obj)
df = pd.DataFrame(col, columns=col_name)
df['location_class'] = -1
df.loc[df['location'] == 'None', 'location_class'] = 0
print(df.dtypes)

df.to_excel('ElonMusk/2023-02-16/user_profile/Profile.xlsx', index=False)
