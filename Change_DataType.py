import pandas as pd



def Change_DataType(df, column_name):

    int_df = df[df[column_name].apply(lambda x: x.is_integer())]

    int_df[column_name] = int_df[column_name].astype(float)
    int_df[column_name] = int_df[column_name].round().astype(int)
    int_df[column_name] = int_df[column_name].round().astype(str)


    float_df = df[~df[column_name].apply(lambda x: x.is_integer())]
    float_df[column_name] = float_df[column_name].astype(str)

    merged_df = pd.concat([int_df, float_df],ignore_index=True)
    
    return merged_df


