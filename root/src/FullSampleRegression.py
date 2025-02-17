# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 01:26:56 2025

@author: Diego
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from tqdm import tqdm
from DataCollect import DataCollect
tqdm.pandas()

class FullSampleRegression(DataCollect):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.full_sample = os.path.join(self.data_path, "FulLSampleRegression")
        if os.path.exists(self.full_sample) == False: os.makedirs(self.full_sample)
        
    def _prep_data(self) -> pd.DataFrame: 
        
        df_out = (self.prep_vix().sort_values(
            "date").
            assign(VIX_shock = lambda x: x.VIX.diff(periods = 3).abs()).
            drop(columns = ["VIX"]).
            merge(right = self.prep_quality(), how = "inner", on = ["date"]).
            dropna())
        
        return df_out
    
    def _get_regression(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_tmp = (df.set_index(
            "date").
            sort_index())
        
        df_out = (sm.OLS(
            endog = df_tmp.value,
            exog  = sm.add_constant(df_tmp.VIX_shock)).
            fit().
            resid.
            to_frame(name = "resid").
            assign(lag_resid = lambda x: x.resid.shift()).
            merge(right = df_tmp, how = "inner", on = ["date"]))
        
        return df_out
    
    def get_full_sample_ols(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.full_sample, "FullSampleOLS.paruqet")
        
        try:
        
            if verbose == True: print("Trying to find Full Sample QMJ data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting it now")
        
            df_out = (self._prep_data().groupby(
                "variable").
                progress_apply(lambda group: self._get_regression(group)).
                drop(columns = ["variable"]).
                reset_index())
            
            if verbose == True: print("Saving data")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
        
def main() -> None:
    
    FullSampleRegression().get_full_sample_ols(verbose = True)
    
if __name__ == "__main__": main()