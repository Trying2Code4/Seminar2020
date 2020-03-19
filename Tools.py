
#ENCODER

# Input:
# Input should be contain (original) columns 'USERID', 'MAILID', 'OFFERID', 'CLICK', 'MailOffer',
#       'USERID_ind', 'OFFERID_ind'
# Ouput:
# formatted: The recoded data in sparse format of [user, item, click]
# key: keychain containing the original row columns and new row comments


def encoder(data,excludeNonclickers=False):    
    if excludeNonclickers:
    # Filter for users that have at least one click
        byUser=data.groupby(["USERID"])
        data=byUser.filter(lambda x: (x["CLICK"].sum()>0) )
        
    # Creating key 
    key = data[['USERID','MailOffer']]
    
    # Creating formatted output
    formatted = data[['USERID_ind','OFFERID_ind','CLICK']]
    formatted.columns = ['user','item','click']
    
    from collections import defaultdict
    #Recoding row indices
    temp = defaultdict(lambda: len(temp))
    formatted['user'] = [temp[ele] for ele in data['USERID']] #overwrite the user column
    #Recoding column indices
    temp = defaultdict(lambda: len(temp))
    formatted['item'] = [temp[ele] for ele in data['MailOffer']] #overwrite the item column
    del temp


    #Creating keys
    key["user"]=formatted["user"]
    key["item"]= formatted["item"]
    
    return formatted, key

#%%
import pandas as pd
import numpy as np
import time

    
def get_test_pred(user,item,p,baseline=0.02192184495444662):
    if np.isnan(user) | np.isnan(item):
        #Prediction will be the baseline in this case
        return baseline
    else:
        user=int(user)
        item=int(item)
        #Retrieve predicted probability from p matrix
        return p[user,item]
    
    

#TEST_PREDICTIONS
# Get predicted probabilities for testset based on probability matrix p        
#Input:
# p = prediction matrix with dimensions, n_users x n_offers
# key = mapping key with the new and original row and column numbers
# test = test dataset in sparse format
# replacement = basic prediction if no prediction available (default is mean overall click rate)
#Output:
# results dataframe, same as testset, but with added column:
# PROBABILITY: predictions for the test dataset
def test_predictions(p, key, trainset, testset, replacement = 0.02192184495444662):
    t0 = time.time()

    results=testset[['USERID', 'MAILID', 'OFFERID', 'CLICK', 'MailOffer', 'ratioU', 'ratioO']]
    # Link new user and item ids back to the original ids
    results=pd.merge(results,key.drop_duplicates(['USERID','user'])[["USERID","user"]]\
                   ,how="left",on=["USERID"])
    results=pd.merge(results,key.drop_duplicates(['MailOffer','item'])[["MailOffer","item"]]\
                   ,how="left",on=["MailOffer"])
    
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

# CV_TEST_RMSE
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
          
    