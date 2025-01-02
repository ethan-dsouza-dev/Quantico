'''
2019 NeurIPS Submission
Title: Time-series Generative Adversarial Networks
Authors: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar

Last Updated Date: May 29th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

Data loading
(1) Load Google dataset
- Transform the raw data to preprocessed data
(2) Generate Sine dataset

Inputs
(1) Google dataset
- Raw data
- seq_length: Sequence Length
(2) Sine dataset
- No: Sample Number
- T_No: Sequence Length
- F_No: Feature Number

Outputs
- time-series preprocessed data
'''

#%% Necessary Packages
import numpy as np
import pickle

#%% Min Max Normalizer

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    pkl_file_path = "normalization_params.pkl"
    with open(pkl_file_path, 'wb') as file:
        last_24 = data[:24, ]
        pickle.dump({"max_val": np.max(last_24, 0), "min_val": np.min(last_24, 0)}, file)
    return numerator / (denominator + 1e-7)

#%% Load Google Data
    
def google_data_loading (seq_length):

    # Load Google Data
    # x = np.loadtxt('data/GOOGLE_BIG.csv', delimiter = ",",skiprows = 1)
    # Loading Apple data
    x = np.loadtxt('data/AAPL_data.csv', delimiter = ",",skiprows = 1)
    # Flip the data to make chronological data
    x = x[::-1]
    # Min-Max Normalizer
    x = MinMaxScaler(x)
    
    # Build dataset
    dataX = []
    
    # Cut data by sequence length
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)
        
    # Mix Data (to make it similar to i.i.d)
    idx = np.random.permutation(len(dataX))
    
    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])
    
    return outputX
  
#%% Sine Data Generation

def sine_data_generation (No, T_No, F_No):
  
    # Initialize the output
    dataX = list()

    # Generate sine data
    for i in range(No):
      
        # Initialize each time-series
        Temp = list()

        # For each feature
        for k in range(F_No):              
                          
            # Randomly drawn frequence and phase
            freq1 = np.random.uniform(0,0.1)            
            phase1 = np.random.uniform(0,0.1)
          
            # Generate Sine Signal
            Temp1 = [np.sin(freq1 * j + phase1) for j in range(T_No)] 
            Temp.append(Temp1)
        
        # Align row/column
        Temp = np.transpose(np.asarray(Temp))
        
        # Normalize to [0,1]
        Temp = (Temp + 1)*0.5
        
        dataX.append(Temp)
                
    return dataX
    
