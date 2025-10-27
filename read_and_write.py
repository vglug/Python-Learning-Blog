import pandas as pd

def read_csv_pandas(filename):
    return pd.read_csv(filename)

def write_csv_pandas(filename,df):
   
    return df.to_csv(filename,index=False,)
def appen_csv_pandas(filename,new_Data):
    df=pd.DataFrame(new_Data)
    df.to_csv(filename, mode="a",index=False,header=False)
def filter_csv_pandas(filename,column,value):
        
        df=pd.read_csv(filename)
        return df[df[column]==value]
if __name__=="__main__":
     data = {
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35],
        "City": ["New York", "London", "Paris"]
    }

df = pd.DataFrame(data)

filename="people.csv"

write_csv_pandas(filename,df)
print(read_csv_pandas(filename))
appen_csv_pandas(filename,{"Name":["parthasarathi"],"Age":[21],"City":["krishnagiri"]})
print(filter_csv_pandas(filename,"Age",25))
