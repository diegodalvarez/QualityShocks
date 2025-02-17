# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 01:58:39 2025

@author: Diego
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from tqdm import tqdm
from FullSampleRegression import FullSampleRegression
tqdm.pandas()

class BootstrappedRegression(FullSampleRegression):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.bootstrapped_path = os.path.join(self.data_path, "BootstrappedOLS")
        if os.path.exists(self.bootstrapped_path) == False: os.makedirs(self.bootstrapped_path)
    
        self.num_samples  = 10_000
        self.sample_size  = 0.3
        self.keep_tickers = ["europe", "global", "global_ex_usa", "pacific", "usa"]
        
    def _bootstrap(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df_tmp = (df.set_index(
            "date").
            sample(frac = self.sample_size))
        
        model = (sm.OLS(
            endog = df_tmp.value,
            exog  = sm.add_constant(df_tmp.VIX_shock)).
            fit())
        
        df_val = (model.params.to_frame(
            name = "val").
            reset_index())
        
        df_out = (model.pvalues.to_frame(
            name = "pval").
            reset_index().
            merge(right = df_val, how = "inner", on = ["index"]))
        
        return df_out
        
    def _bootstrap_ols(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (pd.concat([
            self._bootstrap(df).assign(sim = i + 1)
            for i in tqdm(range(self.num_samples), desc = "Working on {}".format(df.name))]))
        
        return df_out
        
    def bootstrap_ols(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.bootstrapped_path, "BootstrappedRegression.paruqet")
        
        try:
        
            if verbose == True: print("Trying to find Bootstrapped QMJ regresion data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting it now")
            
            df_out = (self._prep_data().query(
                "variable == @self.keep_tickers").
                groupby("variable").
                apply(self._bootstrap_ols).
                reset_index().
                drop(columns = ["level_1"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_bootstrap_sharpe(self, df: pd.DataFrame, df_rtn: pd.DataFrame) -> pd.DataFrame:
        
        variable = df.variable.iloc[0]
        
        param_dict = (df[
            ["index", "val"]].
            set_index("index").
            val.
            to_dict())
        
        df_signal = (df_rtn.sort_values(
            "date").
            query("variable == @variable").
            assign(
                yhat       = lambda x: (param_dict["VIX_shock"] * x.VIX_shock) + param_dict["const"],
                resid      = lambda x: x.value - x.yhat,
                lag_resid  = lambda x: x.resid.shift(),
                signal_rtn = lambda x: np.sign(x.lag_resid) * x.value).
            signal_rtn)
        
        sharpe = df_signal.mean() / df_signal.std() * np.sqrt(252)
        return sharpe
    
    def get_bootsrapped_sharpe(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.bootstrapped_path, "BootstrappedSharpe.paruqet")
        
        try:
        
            if verbose == True: print("Trying to find Bootstrapped QMJ Sharpe data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting it now")
        
            df_tmp = (self.get_full_sample_ols().query(
                "variable == @self.keep_tickers").
                drop(columns = ["resid", "lag_resid"]))
            
            df_out = (self.bootstrap_ols().groupby(
                ["variable", "sim"]).
                progress_apply(lambda group: self._get_bootstrap_sharpe(group, df_tmp)).
                reset_index().
                rename(columns = {0: "sharpe"}))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
        
def main() -> None:
    
    BootstrappedRegression().bootstrap_ols(verbose = True)
    BootstrappedRegression().get_bootsrapped_sharpe(verbose = True)

if __name__ == "__main__": main()