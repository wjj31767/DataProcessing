import pandas as pd
import time
data = pd.read_stata('orbis_financials_spain_cleaned.dta')
for i in data['bvd']:
    pass