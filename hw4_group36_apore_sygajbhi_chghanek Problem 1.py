#!/usr/bin/env python
# coding: utf-8

# In[1]:


# referred from https://towardsdatascience.com/ml-from-scratch-decision-tree-c6444102436a


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder


# In[3]:


dataset = {
    "Age": [24,53,23,25,32,52,22,43,52,48],
    "Salary ($)": [40000,52000,25000,77000,48000,110000,38000,44000,27000,65000],
    "College degree": ['yes','no','no','yes','yes','yes','yes','no','no','yes']
}


# In[4]:


df = pd.DataFrame(dataset)


# In[5]:


df = df.replace({'College degree': {'yes':1,'no':0}})


# In[7]:


plt.scatter(df.iloc[:,0],df.iloc[:,1],c=df.iloc[:,2])
plt.show()


# In[8]:


class DecisionTree():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
         ## initialise the tree 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value


# In[9]:


class DecisionTreeScratch():
    def __init__(self, minimum_split_samples=2, maximum_depth_tree=2):
        # initialise the multi variate tree
        
        self.root = None
        
        # stopping conditions
        self.minimum_split_samples = minimum_split_samples
        self.maximum_depth_tree = maximum_depth_tree
        
    def construct_tree_node(self, df, curr_depth=0):
        # recursive function to build the tree
        
        data, target = df[:,:-1], df[:,-1]
        sample, feat = np.shape(data)
        
        # split until stopping conditions are met
        if sample>=self.minimum_split_samples and curr_depth<=self.maximum_depth_tree:
            # find the best split
            split_best = self.split_best(df, sample, feat)
            # check if information gain is positive
            if split_best["info_gain"]>0:
                # recur left
                left_subtree = self.construct_tree_node(split_best["df_left"], curr_depth+1)
                # recur right
                right_subtree = self.construct_tree_node(split_best["df_right"], curr_depth+1)
                # return decision node
                return DecisionTree(split_best["feature_index"], split_best["threshold"], 
                            left_subtree, right_subtree, split_best["info_gain"])
        
        # compute leaf node
        leafval = self.find_leaf_val(target)
        # return leaf node
        return DecisionTree(value=leafval)
    
    def split_best(self, df, sample, feat):
        # function to get the best split to left and right nodes
        
        # dictionary to store the best split
        split_best = {}
        mig = -float("inf")
        
        # loop over all the features
        for feature_index in range(feat):
            feature_values = df[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for i in possible_thresholds:
                # get current split
                df_left, df_right = self.split(df, feature_index, i)
                # check if childs are not null
                if len(df_left)>0 and len(df_right)>0:
                    y, left_y, right_y = df[:, -1], df_left[:, -1], df_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.find_if(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>mig:
                        split_best["feature_index"] = feature_index
                        split_best["threshold"] = i
                        split_best["df_left"] = df_left
                        split_best["df_right"] = df_right
                        split_best["info_gain"] = curr_info_gain
                        mig = curr_info_gain
                        
        # return best split
        return split_best
    
    def split(self, df, feature_index, threshold):
        # function to split data into its children
        
        df_left = np.array([row for row in df if row[feature_index]<=threshold])
        df_right = np.array([row for row in df if row[feature_index]>threshold])
        return df_left, df_right
    
    def find_if(self, parent, left, right, mode="entropy"):
        # function to find information 
        
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)
        if mode=="gini":
            gain = self.find_gini(parent) - (weight_left*self.find_gini(left) + weight_right*self.find_gini(right))
        else:
            gain = self.entropy(parent) - (weight_left*self.entropy(left) + weight_right*self.entropy(right))
        return gain
    
    def entropy(self, y):
        # function to calculate the entropy
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def find_gini(self, target):
        #function to calculate the gini index
        
        label = np.unique(target)
        g = 0
        for i in label:
            q = len(target[target == i]) / len(target)
            g += q**2
        return 1 - g
    
    def find_leaf_val(self, target):
        # function to find the leaf node values
        
        target = list(target)
        return max(target, key=target.count)
    
    def display(self, decision=None, diff=" "):
        # function to display the tree completely
        
        if not decision:
            decision = self.root

        if decision.value is not None:
            print(decision.value)

        else:
            print("Feature Index"+str(decision.feature_index), "<=", decision.threshold)
            print('Information gain : {}'.format(decision.info_gain))
            print("%sleft child:" % (diff), end="")
            self.display(decision.left, diff + diff)
            print("%sright child:" % (diff), end="")
            self.display(decision.right, diff + diff)
    
    def fit(self, data, target):
        # function to train the model, exaxctly similar to the inbuilt function
        
        df = np.concatenate((data, target), axis=1)
        self.root = self.construct_tree_node(df)
    
    def prediction(self, val):
        #predicting new data 
        
        p = [self.pred(v, self.root) for v in val]
        return p
    
    def pred(self, val, decision):
        # predicting a single data
        
        if decision.value!=None: 
            return decision.value
        value = val[decision.feature_index]
        if value<=decision.threshold:
            return self.pred(val, decision.left)
        else:
            return self.pred(val, decision.right)


# In[10]:


data = df.iloc[:,:-1].values
target = df.iloc[:,-1].values.reshape(-1,1)


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(data , target, test_size=0.3, random_state=42)


# In[13]:


dtc = DecisionTreeScratch(minimum_split_samples=2, maximum_depth_tree=3)
dtc.fit(X_train,y_train)
dtc.display()


# ![download%20%281%29.jfif](attachment:download%20%281%29.jfif)

# ![WhatsApp%20Image%202022-12-09%20at%2011.36.31%20PM.jpeg](attachment:WhatsApp%20Image%202022-12-09%20at%2011.36.31%20PM.jpeg)

# In[ ]:




