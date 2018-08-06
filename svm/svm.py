


'''svm to classify image'''
'''data input has to be dictionary, key is the class name , and value is the image matrix(2D)'''
'''class(dict) to train'''
'''class.predict(img) to do prediction '''
#import and split data set



import numpy as np
import cv2
import os
import random
from matplotlib import pyplot as plt




class imageclass():
    def __init__(self,key,image):
        self.key = key
        self.image = image


    
class svm(object):
    def __init__(self,dict_):
        import logging
        import numpy as np
        import copy
        import time
        dict_ = copy.deepcopy(dict_)
        self.dict = dict_
        if type(self.dict) != dict:
            logging.warning( 'Passed data type is not dictionary, please convert it,we will have the function soon')
            return
        self.classes = len(dict_)
        self.label = list(self.dict.keys())
        self.loss = 0
        try:
            self.dict[list(self.dict.keys())[0]][0].shape[1]
        except:
            self.image_dim = self.dict[list(self.dict.keys())[0]][0].shape[0]
        else:
            logging.warning('image matrix has not been converted to column matrix')
            return
        self.w = np.random.rand(self.classes,self.image_dim)
        for key_ind in range(len(self.dict)):
            for image_ind in range(len(self.dict[list(self.dict.keys())[key_ind]])):
                self.dict[list(self.dict.keys())[key_ind]][image_ind] = imageclass(key_ind,self.dict[list(self.dict.keys())[key_ind]][image_ind])
        start = time.time()
        self.run()
        end = time.time()
        print('Elapsed time {}s'.format(round(end-start,3)))
#evaluate is used for evaluating the loss of current W
    @classmethod
    def convert_to_dict(cls,arr):
        '''yet to develop'''
        return

        
    def evaluate(self):
        from functools import partial
        lf = partial(self.loss_function,delta = 100)
        total_loss = 0
        for key in self.dict:
            for cls in self.dict[key]:
                total_loss += lf(cls.image,cls.key)[0]
        return total_loss
                
#when passing loss function, use functools partial on delta
    def loss_function(self,image,key,delta):
        prod = np.dot(self.w,image)
        loss = 0
        count_fail_meet = 0
        for i in range(len(prod)):
            if i != key:
                if max(0,prod[i]-prod[key] + delta) != 0:
                    count_fail_meet += 1
                    loss += prod[i]-prod[key] + delta
                else:
                    pass
            else:
                pass
        return (loss,count_fail_meet)
            
    def updateW(self):
        step_size = 10e-2
        for key in self.dict:
            for cls in self.dict[key]:
                temp = self.loss_function(cls.image,cls.key,100)
                if temp[0] != 0:
                    for row in range(self.classes):
                        if row != cls.key:
                            self.w[row,:] -= cls.image * step_size
                           
                        else:
                            self.w[row,:] += cls.image * step_size * temp[1]
            
                       
    #update the correct class row with the data vector scale by number of classes which didnt fulfill the margin
    def run(self):
        run = 0
        while self.evaluate() != 0 and run < 1000:
            self.updateW()

    def predict(self,test):
        import logging
        import numpy as np
        '''log = logging.getLogger('name') log.setLevel('INFO')'''
        logging.info('Test data must be resized to one row vector with same length as the training data')
        prediction = []
        if type(test) == np.ndarray:
            assert test.size == self.w.shape[1], 'data input is not in same length'
            test = [test]
        
        if len(test) >1:
            for i in range(len(test)):
                if type(test[i]) == list:
                    assert type(test[i][0]) == np.ndarray and test[i][0].size == self.w.shape[1]
                    test[i] = np.array(test[i][0])
                    print('changed')
                else:
                    assert type(test[i]) == np.ndarray and test[i].size == self.w.shape[1]
                
                
        for arr in test:
            score_prediction = np.dot(self.w,np.transpose(arr))
            prediction.append(self.label[self.find_index_score(score_prediction)])

        return prediction
    
    def find_index_score(self,arr):
        assert len(arr)!=0 ,'Input test data wrong type'
        maxscore = max(arr)
        for i in enumerate(arr):
            if i[1] == maxscore:
                return i[0]
    
    



 
 
 
  
