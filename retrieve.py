import pandas as pd 
import numpy as np
import base64

def retriever(myfile,col,feature):
    data = pd.DataFrame(myfile[feature], columns = [col])

    return data

def download_csv(data):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = data.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="result.csv" target="_blank">Download CSV</a>'

    return href