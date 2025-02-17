# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:04:58 2025

@author: Diego
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS
from   BootstrappedSampleRegression import BootstrappedRegression

class ExpandingOLS(BootstrappedRegression):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.expanding_ols = os.path.join(self.data_path, "ExpandingOLS")
        if os.path.exists(self.expanding_ols) == False: os.makedirs(self.expanding_ols)
        
    def _get_expanding_ols(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df_tmp = (df.set_index(
            "date").
            sort_index())
        
        df_out = (RollingOLS(
            endog     = df_tmp.value,
            exog      = sm.add_constant(df_tmp.VIX_shock),
            expanding = True).
            fit().
            params.
            rename(columns = {
                "const"    : "alpha",
                "VIX_shock": "beta"}).
            dropna().
            merge(right = df, how = "inner", on = ["date"]))
        
        return df_out

        
    def get_expanding_ols(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.expanding_ols, "BootstrappedRegression.paruqet")
        
        try:
        
            if verbose == True: print("Trying to find Expanding QMJ regresion data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting it now")
        
            df_out = (self._prep_data().query(
                "variable == @self.keep_tickers").
                groupby("variable").
                apply(self._get_expanding_ols).
                drop(columns = ["variable"]).
                reset_index().
                drop(columns = ["level_1"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
def main() -> None:
        
    ExpandingOLS().get_expanding_ols(verbose = True)
    
if __name__ == "__main__": main()