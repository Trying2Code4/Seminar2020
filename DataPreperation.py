import pandas as pd
from collections import defaultdict

#Import data
observations = pd.read_csv('Data/Observations_Report.csv', sep=';')

# Convert user ids into row indices
temp = defaultdict(lambda: len(temp))
observations['ROW_IND'] = [temp[ele] for ele in observations['USERID']]
# Convert offer ids into column indices
temp = defaultdict(lambda: len(temp))
observations['COL_IND'] = [temp[ele] for ele in observations['OFFERID']]
del temp

# Create dataset without duplicates (keeping only the last observation of the duplicates)

unique_obs = observations.drop_duplicates(['USERID','OFFERID'], keep='last')
unique_obs.to_csv('Click_master_key.csv', sep = ';', index = FALSE)
unique_obs = unique_obs[['ROW_IND','COL_IND','CLICK']]
unique_obs.to_csv('Click_nodup.csv', sep = ';', index = False)
