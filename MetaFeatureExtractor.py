#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:19:36 2023

@author: mkr01
"""

import pandas as pd
import numpy as np
from pymfe.mfe import MFE

def calculate_mfe(X, y):
    # Select the features to calculate MFE for
    features = ["attr_to_inst","freq_class","inst_to_attr","can_cor","cor","cov",
                "nr_disc","kurtosis","nr_norm","sd_ratio","skewness","w_lambda",
                "attr_ent","class_ent","joint_ent","mut_inf"]
    
    # Convert the values in X and y to integers
    X = X.astype(int)
    y = y.astype(int)
    
    # Initialize the MFE object with selected features
    mfe = MFE(features=features)
    
    # Fit the MFE object to the data
    mfe.fit(X.values, y.values)
    
    # Calculate the MFE
    result = mfe.extract()
    
    # Convert the MFE result to a DataFrame
    mfe_df = pd.DataFrame.from_dict(result, orient='index', columns=['MFE'])
    
    # Convert the MFE values to integers
    mfe_df['MFE'] = mfe_df['MFE'].astype(int)
    
    # Return the MFE DataFrame
    return mfe_df