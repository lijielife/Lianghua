import numpy as np
import pandas as pd
from CkData import cons as ct
from CkData import stock as st


pa_df=st.get_today_all()
# pa_df=st.get_hist_data('600000',start='2017-12-01',end='2018-01-24',ktype='D')
print(pa_df)

