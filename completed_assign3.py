__author__ = 'Richard'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import collections
from sklearn.cross_validation import train_test_split
from sklearn import linear_model

# this function takes the drugcount dataframe as input and output a tuple of 3 data frames: DrugCount_Y1,DrugCount_Y2,DrugCount_Y3
def process_DrugCount(drugcount):

    drug_count_1 = []
    drug_count_2 = []
    drug_count_3 = []
    for item in drugcount.values:
        item[3] = int(item[3].replace('+', ''))
        lis = []
        if item[1] == 'Y1':
            lis.extend([item[0], item[2], item[3]])
            drug_count_1.append(lis)
        elif item[1] == 'Y2':
            lis.extend([item[0], item[2], item[3]])
            drug_count_2.append(lis)
        elif item[1] == 'Y3':
            lis.extend([item[0], item[2], item[3]])
            drug_count_3.append(lis)

    f = lambda drugs : pd.DataFrame({'DrugCount': [item[2] for item in drugs],
                                     'MemberID': [item[0] for item in drugs],
                                     'DSFS': [item[1] for item in drugs]})

    DrugCount_Y1 = f(drug_count_1)
    DrugCount_Y2 = f(drug_count_2)
    DrugCount_Y3 = f(drug_count_3)

    s = lambda drug_count : drug_count.columns.tolist()
    a = lambda cols : [cols[2], cols[0], cols[1]]

    cols1 = a(s(DrugCount_Y1))
    cols2 = a(s(DrugCount_Y2))
    cols3 = a(s(DrugCount_Y3))

    DrugCount_Y1 = DrugCount_Y1[cols1]
    DrugCount_Y2 = DrugCount_Y2[cols2]
    DrugCount_Y3 = DrugCount_Y3[cols3]

    return (DrugCount_Y1,DrugCount_Y2,DrugCount_Y3)

# this function converts strings such as "1- 2 month" to "1_2"
def replaceMonth(string):
    a_new_string = string.replace(' ', '').replace('-', '_').replace('months', '').replace('month', '')
    return a_new_string

# this function processes a yearly drug count data
def process_yearly_DrugCount(aframe):

    aframe['DSFS'] = aframe['DSFS'].apply(replaceMonth)

    dummy_var = pd.get_dummies(aframe['DSFS'], prefix='DSFS').sort_index()

    concat = pd.concat([aframe, dummy_var], axis=1)
    concat = concat.drop('DSFS', 1)

    grouped = concat.groupby(concat['MemberID'], as_index=False).sum()

    s = lambda drug_count : drug_count.columns.tolist()
    a = lambda cols : [cols[0], cols[1], cols[2], cols[5], cols[6], cols[7], cols[8],
                       cols[9], cols[10], cols[11], cols[12], cols[13], cols[3], cols[4]]

    k = a(s(grouped))
    grouped = grouped[k]
    processed_frame = grouped.rename(columns = {'DrugCount':'Total_DrugCount'})

    return processed_frame


# this is the function to split training dataset to training and test.
def split_train_test(arr, test_size=.3):
    train, test = train_test_split(arr, test_size=0.33)
    train_X = train[:, :-1]
    #print train_X.shape
    train_y = train[:, -1]
    #print train_y.shape
    test_X = test[:,:-1]
    test_y = test[:, -1]
    return (train_X,train_y,test_X,test_y)

# run linear regression.
def linear_regression((train_X,train_y,test_X,test_y)):
    regr = linear_model.LinearRegression()
    regr.fit(train_X, train_y)
    print 'Coefficients: \n', regr.coef_
    pred_y = regr.predict(test_X) # your predicted y values
    # The root mean square error
    mse = np.mean((pred_y - test_y) ** 2)
    import math
    rmse = math.sqrt(mse)
    print ("RMSE: %.2f" % rmse)
    from sklearn.metrics import r2_score
    r2 = r2_score(pred_y, test_y)
    print ("R2 value: %.2f" % r2)

# for a real-valued variable, replace missing with median
def process_missing_numeric(df, variable):

    df['DrugCount_Missing'] = np.where(df['Total_DrugCount'].isnull(),1,0)
    medianDrugcount = df.Total_DrugCount.median()
    df.Total_DrugCount.fillna(medianDrugcount, inplace=True)

    return

# This function prints the ratio of missing values for each variable.
def print_missing_variables(df):
    for variable in df.columns.tolist():
        percent = float(sum(df[variable].isnull()))/len(df.index)
        print variable+":", percent

def main():
    pd.options.mode.chained_assignment = None # remove the warning messages regarding chained assignment.
    daysinhospital = pd.read_csv('DaysInHospital_Y2.csv')
    drugcount = pd.read_csv('DrugCount.csv')
    li = map(process_yearly_DrugCount, process_DrugCount(drugcount))
    DrugCount_Y1_New = li[0]
    Master_Assn1 = pd.merge(daysinhospital, DrugCount_Y1_New, how='left', on='MemberID')# your code here to create Master_Assn1 by merging daysinhospital and DrugCount_Y1_New
    newColumns = ['MemberID', 'ClaimsTruncated', 'Total_DrugCount', 'DSFS_0_1', 'DSFS_1_2', 'DSFS_2_3', 'DSFS_3_4', 'DSFS_4_5', 'DSFS_5_6', 'DSFS_6_7', 'DSFS_7_8', 'DSFS_8_9', 'DSFS_9_10', 'DSFS_10_11', 'DSFS_11_12','DaysInHospital']
    Master_Assn1 = Master_Assn1[newColumns]
    '''outputs:
     MemberID  ClaimsTruncated  Total_DrugCount  DSFS_0_1  DSFS_1_2  DSFS_2_3  \
0  24027423                0                3         0         0         0
1  98324177                0                1         1         0         0
2  33899367                1               23         1         0         1

   DSFS_3_4  DSFS_4_5  DSFS_5_6  DSFS_6_7  DSFS_7_8  DSFS_8_9  DSFS_9_10  \
0         1         0         0         0         0         0          0
1         0         0         0         0         0         0          0
2         1         1         1         1         1         1          1

   DSFS_10_11  DSFS_11_12  DaysInHospital
0           0           0               0
1           0           0               0
2           1           0               1
    '''
    process_missing_numeric(Master_Assn1, 'Total_DrugCount')
    Master_Assn1.fillna(0, inplace=True)
    Master_Assn1.drop('MemberID', axis = 1, inplace=True)
    print Master_Assn1.head()
    arr = Master_Assn1.values
    linear_regression(split_train_test(arr))


if __name__ == '__main__':
    main()




