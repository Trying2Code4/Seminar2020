#Encoder

# Input:
# data: Data with rows and columns which need to be reordered
# Input should contain column names [ROW_IND, COL_IND, CLICK]
# Ouput:
# formatted: The recoded data in the form of [user, item, count]
# key: keychain containing the original row columns and new row comments
def encoder(data):
    from collections import defaultdict

    #Recoding row indices
    temp = defaultdict(lambda: len(temp))
    data['user'] = [temp[ele] for ele in data['ROW_IND']]

    #Recoding column indices
    temp = defaultdict(lambda: len(temp))
    data['item'] = [temp[ele] for ele in data['COL_IND']]
    del temp

    #Key output formulation
    key = data[['ROW_IND','COL_IND','user', 'item']]

    #Reformatting output
    formatted = data[['user', 'item', 'CLICK']]
    formatted.columns = ['user', 'item', 'count']

    return formatted, key

#test_predictions
#Input:
# p = prediction matrix with dimensions, users x products
# key = mapping key with the new and original row and column numbers
# test = test dataset in sparse format
# replacement = basic prediction if no prediction available (defaulted at mean overall click rate)
#Output:
# predict: predictions for the test dataset

def test_predictions(p, key, test, replacement = 0.021956):
    #Define iterating variable
    i=0

    #Defining output
    output = test

    #Fill the output column with prediction with the mean overall click rate
    output['prediction'] = replacement

    #Looping over the test matrix
    for d in test.values:
        #Selecting the user and item combination from the sparse test data set
        testuser = int(d[0])
        testitem = int(d[1])

        #Check if the test value occurs in the training matrix (of user x item combinations)
        if key[(key["ROW_IND"]==testuser) & (key["COL_IND"] == testitem)].empty:
            #nothing
            pass
        else:
            #retrieve predicted probability
            user = int(key[(key["ROW_IND"]==testuser) & (key["COL_IND"] == testitem)]['user'])
            item = int(key[(key["ROW_IND"]==testuser) & (key["COL_IND"] == testitem)]['item'])
            output['prediction'][i] = p[user][item]
        i+=1
        print(i)

    #return output
    return output


#