import pandas as pd
import numpy as np

# To handle some example data
# Since the age in demographics.csv is indicating the age group rather than the actual age,
# this script is to make up some actual age in order to make the data usable for epios
df = pd.read_csv('demographics.csv')
df.loc[df['age_group'] < 16, 'age_group'] = df['age_group'][df['age_group'] < 16].apply(lambda x: x * 5 + np.random.randint(0, 5)) # noqa
df.loc[df['age_group'] == 16, 'age_group'] = df['age_group'][df['age_group'] == 16].apply(lambda x: x * 5 + np.random.randint(0, 20)) # noqa

df.to_csv('demographics_processed.csv')
