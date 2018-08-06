import numpy as np
from matplotlib import pyplot as plt
import math
from functools import reduce




class Sobel:
    def __init__(self,img):
        self.img = img
        self.size = img.size
        self.shape = img.shape
        self.gx = [0]*self.size
        self.gy = [0]*self.size
        self.transform()
        h,w = self.shape
        w_mod = w%8
        h_mod = h%8
        self.cell_size = 8
        self.n_bucket = 9
        self.block_size = 2
        self.gx1_trim = self.gx1[0:(h-h_mod),0:(w-w_mod)]
        self.gy1_trim = self.gy1[0:(h-h_mod),0:(w-w_mod)]
        assert self.gx1_trim.size%self.cell_size ==0
        
    def transform(self):
        vec = np.reshape(self.img,-1)
        n = self.shape[1]
        total = self.size
        for i in range(total):
            if i<n or i>=total-n or (i+1)%n==0 or i%n == 0:
                self.gx[i] = vec[i]
                self.gy[i] = vec[i]
            else:
                assert i-n >0
                self.gx[i] = 2*vec[i+1] - 2*vec[i-1] + vec[i-n+1] - vec[i-n-1] + vec[i+n+1] - vec[i+n-1]
                self.gy[i] = 2*vec[i+n]- 2*vec[i-n] + vec[i+n-1] - vec[i-n-1]  + vec[i+n+1]-vec[i-n+1]
        self.gx1 = np.reshape(self.gx,self.shape)
        self.gy1 = np.reshape(self.gy,self.shape)

    def sobelx(self):
        plt.figure(figsize=(7,7))
        plt.gray()
        plt.imshow(self.gx1)
    
    def sobely(self):
        plt.figure(figsize=(7,7))
        plt.gray()
        plt.imshow(self.gy1)
        
    def sobel(self):
        g = np.sqrt(np.array(self.gx)**2 + np.array(self.gy)**2)
        g1 = np.reshape(g,self.shape)
        plt.figure(figsize=(7,7))
        plt.gray()
        plt.imshow(g1)
    

        
            
class HOG(Sobel):
    def assign_bucket(self,m,direc,bucket_vals):    
        if math.isnan(direc):
            direc = 90
        left_bin = int(direc/20.0)
        right_bin = (int(direc/20.0)+1)%self.n_bucket
        #assert 0 <= left_bin <= right_bin <= n_bucket
        left_val = m * ((left_bin+1)*20 - direc)/20
        right_val = m * (direc - left_bin*20)/20 
        #print(left_val,right_val,direc,m)
        assert left_val>=0 and right_val >=0 and direc>=0
        bucket_vals[left_bin] += left_val
        bucket_vals[right_bin] += right_val
        
        
    def get_cell_hist(self,loc_x,loc_y):
        cell_x = self.gx1_trim[loc_x:loc_x + self.cell_size,loc_y:loc_y+self.cell_size]
        cell_y = self.gy1_trim[loc_x:loc_x + self.cell_size,loc_y:loc_y+self.cell_size]
        magnitude = np.sqrt(np.array(cell_x)**2 + np.array(cell_y)**2)
        direction =  np.abs(np.arctan(cell_y/cell_x)*180/np.pi)
        bucket_vals = np.zeros(self.n_bucket)
    #    map(    lambda m: assign_bucket(m[0],m[1],bucket_vals),
    #            list(zip(magnitude.flatten(),direction.flatten()))
    #        )
        for (m,d) in zip(magnitude.flatten(),direction.flatten()):
            self.assign_bucket(m,d,bucket_vals)
        return bucket_vals
    
    
    
    def get_block_hist(self,loc_x,loc_y):
        return(
            reduce(
                lambda arr1,arr2: np.concatenate((arr1,arr2),axis=0),
                [self.get_cell_hist(x,y) for (x,y) in zip([loc_x,loc_x+self.cell_size,loc_x,loc_x+self.cell_size],
                 [loc_y,loc_y,loc_y+self.cell_size,loc_y+self.cell_size])])
            )
    
    
    #go through the starting point of every block and concat all     
    def overlap_block(self):
        row,column = self.shape
        overlap_arr = []
        for i in range(self.size):
            if i%self.cell_size  == 0 and math.floor(i/column)%self.cell_size ==0:
                cord_row = math.floor(i/column)
                cord_col = i%column
                overlap_arr.append(self.get_block_hist(cord_row,cord_col))
                
        return overlap_arr
        
    
  


