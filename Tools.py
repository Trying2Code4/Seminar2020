#ENCODER

# Input:
# data: Data with rows and columns which need to be reordered
# Input should be contain (original) column names [USERID, OFFERID, CLICK]
# Ouput:
# formatted: The recoded data in sparse format of [user, item, click]
# key: keychain containing the original row columns and new row comments


def encoder(data):
    from collections import defaultdict
    #Key output formulation
    key = data[['USERID','OFFERID']]

    #Recoding row indices
    temp = defaultdict(lambda: len(temp))
    data['user'] = [temp[ele] for ele in data['USERID']]
    #Recoding column indices
    temp = defaultdict(lambda: len(temp))
    data['item'] = [temp[ele] for ele in data['OFFERID']]
    del temp

    #Creating formatted output
    formatted = data[['user', 'item', 'CLICK']]
    formatted.columns = ['user', 'item', 'click']
    
    #Creating keys
    key["user"]=data["user"]
    key["item"]= data["item"]
    
    return formatted, key

def test_encoder(test,key):

    
    return test

#%%
import pandas as pd
import numpy as np
import time
import swifter

#TEST_PREDICTIONS
#Input:
# p = prediction matrix with dimensions, n_users x n_offers
# key = mapping key with the new and original row and column numbers
# test = test dataset in sparse format
# replacement = basic prediction if no prediction available (default is mean overall click rate)
#Output:
# results dataframe, same as testset, but with added column:
# PROBABILITY: predictions for the test dataset
def test_predictions(p, key, train, test, replacement = 0.02192184495444662):
    t0 = time.time()
    #iterating variable
    iter=0 
    #Define variables for process status
    j=1
    total=len(test)
    
    #Defining output
    output = test.copy()

    #Fill the output column with prediction with the mean overall click rate
    output['PROBABILITY'] = replacement

    #Looping over the test matrix
    for index,row in test.iterrows():
        #Selecting the user and item combination from the sparse test data set
        testuser = int(row[0]) #USERID
        testitem = int(row[2]) #OFFERID

        # Check if the user and the item both occur in the training matrix
        if (key[(key["USERID"]==testuser)].empty) | (key[(key["OFFERID"] == testitem)].empty):
            #Prediction will be the baseline in this case
            pass
        # Predict 0 for users that haven't clicked on anything in the training set
        elif sum(train.loc[train["USERID"]==testuser,"CLICK"])<1:
            output.loc[index,'PROBABILITY']=0
        else:
            #Find new indexes as they appear in the reformatted (training) matrix
            trainuser=key[key["USERID"] == testuser]["user"].values[0]
            trainitem=key[key["OFFERID"] == testitem]["item"].values[0]
            #Retrieve predicted probability from p matrix
            output.loc[index,'PROBABILITY'] = p[trainuser][trainitem]
            
        iter+=1
        if iter/total > (j*0.1):    
            print('The process is', (j*10), '% ready')
            j+=1
        
    t1=time.time()
    print('The process is 100 % ready. Time elapsed: ', round(t1-t0), ' seconds')  
    
    return output


    
def get_test_pred(user,item,p,baseline=0.02192184495444662):
    if np.isnan(user) | np.isnan(item):
        #Prediction will be the baseline in this case
        return baseline
        # Predict 0 for users that haven't clicked on anything in the training set
    #elif sum(train.loc[train["USERID"]==userid,"CLICK"])<1:
    #    return 0
    else:
        user=int(user)
        item=int(item)
        #Retrieve predicted probability from p matrix
        return p[user,item]
    
    
def test_predictions2(p, key, trainset, testset, replacement = 0.02192184495444662):
    t0 = time.time()

    # Link new user and item ids back to the original ids
    results=pd.merge(testset,key.drop_duplicates(['USERID','user'])[["USERID","user"]]\
                   ,how="left",on=["USERID"])
    results=pd.merge(results,key.drop_duplicates(['OFFERID','item'])[["OFFERID","item"]]\
                   ,how="left",on=["OFFERID"])
    print("Getting predictions...")
    # Get predictions
    results["PROBABILITY"]=results.apply(lambda row: get_test_pred(row['user'],row['item'],p,replacement), axis=1)
    print("Overriding some  predictions..")
    
    # Override predictions for non-clickers
    byUser=trainset.groupby(["USERID"])
    #users (id's) that never clicked on anything:
    nonclickers=byUser.filter(lambda x: (x["CLICK"].sum()<1) )["USERID"].unique()
    results[results["USERID"].isin(nonclickers)]["PROBABILITY"]=0
        
    ## !
    t1=time.time()
    print('The process is ready. Time elapsed: ', round(t1-t0), ' seconds')
    return results
    

    
#%%

#CV_TEST_PREDICTIONS
# Get test predictions for various model prediction matrices
#Input:
# list_P = list of prediction matrices with dimensions, n_users x n_offers
# key = mapping key with the new and original row and column numbers
# test = test dataset in sparse format
# replacement = basic prediction if no prediction available (default is mean overall click rate)
#Output:
# results dataframe, same as testset, but with added columns:
# PROBABILITY_1,PROBABILITY_2,PROBABILITY_3: predictions for the test dataset for prediction matrix 1,2,3..
def CV_test_RMSE(list_P, key, train,test, replacement = 0.02192184495444662):
    import numpy as np
    import pandas as pd
    #iterating variable
    iter=0 
    
    #Defining dataframe to store results
    df = test.copy()
        
    #Defining output
    num_methods=len(list_P)
    output=np.zeros(num_methods)

    #Fill the new output columns with prediction with the mean overall click rate
    method_iter=1
    for P in list_P:
        df['PROBABILITY_'+str(method_iter)] = replacement
        method_iter+=1

    #Looping over the test matrix
    for index,row in test.iterrows():
        #Selecting the user and item combination from the sparse test data set
        testuser = int(row[0]) #USERID
        testitem = int(row[2]) #OFFERID

        #Check if the user and the item both occur in the training matri
        if (key[(key["USERID"]==testuser)].empty) | (key[(key["OFFERID"] == testitem)].empty):
            #Prediction will be the baseline in this case
            pass
        elif sum(train.loc[train["USERID"]==testuser,"CLICK"])<1:
            df.loc[index,'PROBABILITY_'+str(method_iter)]=0
        else:
            #user = int(key[(key["USERID"]==testuser) & (key["OFFERID"] == testitem)]['user'])
            #item = int(key[(key["USERID"]==testuser) & (key["OFFERID"] == testitem)]['item'])
            #Find new indexes as they appear in the reformatted (training) matrix
            trainuser=key[key["USERID"] == testuser]["user"].values[0]
            trainitem=key[key["OFFERID"] == testitem]["item"].values[0]
            
            #Retrieve predicted probability from each p matrix
            method_iter=1
            for P in list_P:
                df.loc[index,'PROBABILITY_'+str(method_iter)] = P[trainuser][trainitem]
                method_iter+=1
        iter+=1
    
    # Getting RMSEs for each methods

    for c in range(1,num_methods+1):
        p= df.loc[:,'PROBABILITY_'+str(c)]
        click=df["CLICK"]
        e=click-p
        e*=e
        RMSE=np.mean(e)
        RMSE=np.power(RMSE,0.5)
        output[(c-1)]=RMSE
        
    return output
        
        
def CV_test_RMSE2(list_P, key, train,test, replacement = 0.02192184495444662):
    import numpy as np
    import pandas as pd
    
    #Defining dataframe to store probabilities
    df = test.copy()
    
    # Finding the users in the training set that haven't clicked
    byUser=train.groupby(["USERID"])
    #users (id's) that never clicked on anything:
    nonclickers=byUser.filter(lambda x: (x["CLICK"].sum()<1) )["USERID"].unique()
        
    #Defining output
    num_methods=len(list_P)
    output=np.zeros(num_methods)

    #Link the dataframe to original user and offer keys
    df=pd.merge(df,key.drop_duplicates(['USERID','user'])[["USERID","user"]]\
                   ,how="left",on=["USERID"])
    df=pd.merge(df,key.drop_duplicates(['OFFERID','item'])[["OFFERID","item"]]\
                   ,how="left",on=["OFFERID"])
        
    # Get predictions for all methods
    method_iter=1
    for P in list_P:
        # Getting predictions from prob matrix / baseline
        df['PROBABILITY_'+str(method_iter)] = df.apply(lambda row: get_test_pred(row['user'],row['item'],P,replacement), axis=1)
        method_iter+=1   
        # Override predictions for non-clickers
        df[df["USERID"].isin(nonclickers)]['PROBABILITY_'+str(method_iter)]=0
    
    # Getting RMSEs for each methods
    click=df["CLICK"]
    for c in range(1,num_methods+1):
        probs = df.loc[:,'PROBABILITY_'+str(c)]
        e=click-probs
        e*=e
        RMSE=np.mean(e)
        RMSE=np.power(RMSE,0.5)
        output[(c-1)]=RMSE
        
    return output
        
#%% TRAIN AND TEST SET GENERATION
def train_test(nObs=100000,testSize=0.2,seed=1):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # LOAD DATA
    observations = pd.read_csv('Observations_Report.csv', sep=';')
    # MAKE A SMALLER SUBSET AND TRAIN + TEST
    observationsSmall = observations.sort_values(by=['USERID'], axis = 0, ascending = True)[1:nObs]
    ## Train test split
    trainset, testset = train_test_split(observationsSmall, test_size = testSize, random_state=seed)
    return trainset,testset    
    