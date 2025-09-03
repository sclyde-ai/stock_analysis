import numpy as np
import pandas as pd

def get_values_ranking(matrix, reverse=True):
    np_matrix = np.array(matrix)
    
    rows, cols = np_matrix.shape
    elements_with_positions = []
    
    for i in range(rows):
        for j in range(cols):
            elements_with_positions.append((np_matrix[i, j], (i, j)))
    
    sorted_elements = sorted(elements_with_positions, key=lambda x: x[0], reverse=reverse)
    
    return sorted_elements

def get_values_ranking_df(matrix, ascending=False):
    np_matrix = np.array(matrix)
    rows, cols = np_matrix.shape
    
    data = []
    for i in range(rows):
        for j in range(cols):
            data.append({
                'value': np_matrix[i, j],
                'row': i,
                'column': j,
                'index': i*cols + j
            })
    
    df = pd.DataFrame(data)
    df = df.set_index('index')
    df_sorted = df.sort_values(by='value', ascending=ascending).reset_index(drop=True)
    
    return df_sorted