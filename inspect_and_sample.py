import pandas as pd
import ast

df = pd.read_csv('ptbxl_database.csv')
scp_statements = pd.read_csv('scp_statements.csv', index_col=0)

def get_superclasses(scp_codes_str):
    try:
        codes = ast.literal_eval(scp_codes_str)
    except:
        return []
    superclasses = set()
    for code, val in codes.items():
        if val > 0: # confidence > 0
            if code in scp_statements.index:
                scls = scp_statements.loc[code, 'diagnostic_class']
                if pd.notna(scls):
                    superclasses.add(scls)
    return list(superclasses)

df['superclasses'] = df['scp_codes'].apply(get_superclasses)

# Let's drop duplicate patient_id to be safe
df_unique = df.drop_duplicates(subset=['patient_id']).copy()

def has_superclass(sups, target):
    return target in sups

print("Total patients with unique records:", len(df_unique))

# Let's count records with strictly NORM only
norm_only = df_unique[df_unique['superclasses'].apply(lambda x: x == ['NORM'])]
print("NORM only count:", len(norm_only))

# Let's count records with exactly one of the abnormal classes and no NORM
mi_only = df_unique[df_unique['superclasses'].apply(lambda x: len(x) == 1 and 'MI' in x)]
sttc_only = df_unique[df_unique['superclasses'].apply(lambda x: len(x) == 1 and 'STTC' in x)]
cd_only = df_unique[df_unique['superclasses'].apply(lambda x: len(x) == 1 and 'CD' in x)]
hyp_only = df_unique[df_unique['superclasses'].apply(lambda x: len(x) == 1 and 'HYP' in x)]

print("MI only count:", len(mi_only))
print("STTC only count:", len(sttc_only))
print("CD only count:", len(cd_only))
print("HYP only count:", len(hyp_only))
