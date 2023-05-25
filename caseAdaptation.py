#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:31:53 2023

@author: mkr01
"""

# Step : Case adaptation and modification
new_instances = [
    [25, "Male"],  # Example new instance 1
    [40, "Female"]  # Example new instance 2
]

# Encode categorical features of new instances
categorical_features = [1]  # Specify the indices of the categorical features
encoder = OrdinalEncoder()
new_instances_encoded = encoder.fit_transform(new_instances)
new_instances_encoded = new_instances_encoded.astype(int)  # Convert to integer

modified_cases = []
for instance_encoded in new_instances_encoded:
    meta_feature_new = instance_encoded  # Extract meta features from the new instance

    distances, indices = nn_model.kneighbors([meta_feature_new])
    similar_cases = X.iloc[indices.flatten()]  # Retrieve similar cases based on meta features

    for _, case in similar_cases.iterrows():
        # Apply your bias reduction technique to modify the case
        modified_case = case.copy()  # Create a copy of the case
        
        # Calculate skewness of numerical features
        numerical_features = ["age", "hours-per-week"]
        skewness = case[numerical_features].apply(lambda x: skew(x))
        
        # Example: Adjust the age based on skewness
        if skewness["age"] > 0:
            modified_case["age"] += 5
        else:
            modified_case["age"] -= 5
        
        modified_cases.append(modified_case)

# Convert modified cases to a DataFrame for further processing
modified_cases = pd.DataFrame(modified_cases)

# Use modified_cases for further processing or making predictions