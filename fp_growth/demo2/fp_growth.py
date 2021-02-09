import pandas as pd
import pyfpgrowth

df = pd.DataFrame([
    'A,B,E',
    'B,D',
    'B,C',
    'A,B,D',
    'A,C',
    'B,C',
    'A,C',
    'A,B,C,E',
    'A,B,C'
])

df = pd.DataFrame(['bread,milk,vegetable,fruit,eggs',
               'noodle,beef,pork,water,socks,gloves,shoes,rice',
               'socks,gloves',
               'bread,milk,shoes,socks,eggs',
               'socks,shoes,sweater,cap,milk,vegetable,gloves',
               'eggs,bread,milk,fish,crab,shrimp,rice'])

print(df)
df.columns = ['items']
df['items'] = df['items'].map(lambda x: x.split(','))

print(df)

res = []
for i in df.values.tolist():
    print(i[0])
    res.append(i[0])



patterns = pyfpgrowth.find_frequent_patterns(res, 2)
print(patterns)

rules = pyfpgrowth.generate_association_rules(patterns, 0.7)
print(rules)
