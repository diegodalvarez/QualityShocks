# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 23:07:24 2025

@author: Diego
"""

import os
import pandas as pd
import yfinance as yf

class DataCollect: 
    
    def __init__(self) -> None:
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_path  = os.path.abspath(os.path.join(self.script_dir, os.pardir))
        self.repo_path  = os.path.abspath(os.path.join(self.root_path, os.pardir))
        self.data_path  = os.path.join(self.repo_path, "data")
        self.raw_path   = os.path.join(self.data_path, "RawData")
        self.prep_path  = os.path.join(self.data_path, "PrepData")
        
        if os.path.exists(self.data_path) == False: os.makedirs(self.data_path)
        if os.path.exists(self.raw_path) == False: os.makedirs(self.raw_path)
        if os.path.exists(self.prep_path) == False: os.makedirs(self.prep_path)
        
        self.bbg_path = r"C:\Users\Diego\Desktop\app_prod\BBGData\data"
        
    def get_vol(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_path, "RawVIX.parquet")
        
        try:
        
            if verbose == True: print("Trying to find raw VIX data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting it now")
            df_out = yf.download(tickers = ["^VIX"])
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
            
    def prep_vix(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.prep_path, "PrepVIX.parquet")
        
        try:
        
            if verbose == True: print("Trying to find prepped VIX data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting it now")
            
            df_out = (self.get_vol()[
                ["Adj Close"]].
                reset_index().
                rename(columns = {
                    "Adj Close": "VIX",
                    "Date"     : "date"}).
                assign(date = lambda x: pd.to_datetime(x.date).dt.date))
        
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_qmj(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.prep_path, "PrepQMJ.parquet")
        
        try:
        
            if verbose == True: print("Trying to find prepped QMJ data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting it now")
        
            read_in_path = os.path.join(self.raw_path, "Quality Minus Junk Factors Daily.xlsx")
            df_out       = (pd.read_excel(
                io         = read_in_path,
                sheet_name = "QMJ Factors",
                skiprows   = 18).
                rename(columns = {"DATE": "date"}).
                assign(date = lambda x: pd.to_datetime(x.date).dt.date).
                melt(id_vars = "date").
                assign(variable = lambda x: x.variable.str.lower().str.replace(" ", "_")).
                dropna())
        
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_jpm(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.prep_path, "PrepJPMQuality.parquet")
        
        try:
        
            if verbose == True: print("Trying to find prepped JPMorgan Quality data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting it now")
        
            paths = [
                os.path.join(self.bbg_path, "JPQWIN.parquet"),
                os.path.join(self.bbg_path, "JPQLAG.parquet")]
            
            df_out = (pd.read_parquet(
                path = paths, engine = "pyarrow").
                assign(security = lambda x: x.security.str.split(" ").str[0]).
                pivot(index = "date", columns = "security", values = "value").
                pct_change().
                assign(
                    value    = lambda x: x.JPQWIN - x.JPQLAG,
                    variable = "quality")
                [["value", "variable"]].
                dropna().
                reset_index())
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_bloomberg(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.prep_path, "PrepBloombergFactors.parquet")
        
        try:
        
            if verbose == True: print("Trying to find prepped JPMorgan Quality data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Saving data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting it now")
            
            paths = [
                os.path.join(self.bbg_path, "PEARNVUS.parquet"),
                os.path.join(self.bbg_path, "PLEVERUS.parquet"),
                os.path.join(self.bbg_path, "PPROFTUS.parquet")]
            
            df_out = (pd.read_parquet(
                path = paths, engine = "pyarrow").
                assign(security = lambda x: x.security.str.split(" ").str[0]).
                pivot(index = "date", columns = "security", values = "value").
                pct_change().
                reset_index().
                melt(id_vars = "date").
                rename(columns = {"security": "variable"}).
                dropna())
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out

def main() -> None:
        
    DataCollect().get_vol(verbose = True)
    DataCollect().prep_vix(verbose = True)
    DataCollect()._get_qmj(verbose = True)
    DataCollect()._get_jpm(verbose = True)
    DataCollect()._get_bloomberg(verbose = True)
    
if __name__ == "__main__": main()