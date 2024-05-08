import sys, numpy as np, pandas as pd, os
sys.path.append('/Users/kingsbury/work/NSD/')
from numpy import array, hstack, round as npround, var, argsort
from scsd import *

data_path = '/Users/kingsbury/work/data/scsd_databases/'
html_path = '/Users/kingsbury/work/data/htmls/'
dfs_path = '/Users/kingsbury/work/NSD/data/scsd/'
output_path = '/Users/kingsbury/work/NSD/data/scsd/'

all_motif = []
to_sqlite = []
ssr_html = '<h2>Available CCDC structures by refcode for SCSD <br> Some examples may appear in multiple databases - contact C.K. for more info </h2>'    

for name, model in model_objs_dict.items():
    if isinstance(model.database_path, str):
        df = pd.read_pickle(output_path + model.database_path)
        print(name)
        for_all = [[name, refcode] for refcode in df['name'].values.tolist()]
        #adding an index lookup
        ssr_html = ssr_html + '<h2>{}</h2> \n <p>{}</p> \n <br>'.format(name, ', '.join(["<a href = '/scsd/{0}'>{0}</a>".format(x) for x in df['name'].values.tolist()]))
        all_motif.append(for_all)

       #df2 = pd.read_pickle(output_path + model.database_path)
       #if 'nearest' in df.columns:
       #    for_all = [[name, refcode, coords, nearest] for refcode, coords, nearest in df[['name','coords','nearest']].values.tolist()]
       #else:
       #    for_all = [[name, refcode, coords, None] for refcode, coords in df[['name','coords']].values.tolist()]
       #to_sqlite.append(for_all)
        

df_out = pd.DataFrame([item for sublist in all_motif for item in sublist])
df_out.columns = ['df_name', 'name']
df_out.to_pickle(output_path + 'combined_df.pkl')

#df_out2 = pd.DataFrame([item for sublist in to_sqlite for item in sublist])
#df_out2.columns = ['name', 'model_name', 'coords', 'nearest']
#
#import sqlite3
#conn = sqlite3.connect(output_path + 'all_scsd.sqlite')
#df_out2.to_sql('all',conn,if_exists='replace', index=False)

f = open('/Users/kingsbury/work/NSD/static/scsd_structure_refcodes.html','w')
f.write(ssr_html)
f.close()
