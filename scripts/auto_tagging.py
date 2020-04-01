#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:05:34 2020

@author: ron
"""

import os
from os.path import join
import shutil
import numpy as np

def duplicate_label(start, stop,  file_name_to_duplicate, path_):
    source = join(path_, file_name_to_duplicate)
    for i in range(start, stop):
        
        file_dest = str(i).zfill(10) + ".txt"
        dest_path = join(path_, file_dest)
        print("Copy {source} to {dest_path}", source, dest_path)
        shutil.copy(source, dest_path)

def duplicate_label_index(path_, index, dup):
    source = join(path_,  str(index).zfill(10) + ".txt")
    for i in range(index +1 , index + dup):

        file_dest = str(i).zfill(10) + ".txt"
        dest_path = join(path_, file_dest)
        print("Copy {source} to {dest_path}", source, dest_path)
        shutil.copy(source, dest_path)
        
def remove_size_mutiple(path_, index, dup):
    for i in range(index +1 , index + dup):
        file_dest = str(i).zfill(10) + ".txt"
        dest_path = join(path_, file_dest)
        remove_size_and_save(dest_path)

def remove_size_and_save(dest_path):
    arr = remove_side(dest_path)
    print("dest_path", dest_path)


    if arr.shape == (0,):
        np.savetxt(dest_path, np.array(arr))
    else:
        if len(arr.shape) == 1:
            arr = np.expand_dims(arr, axis=0)
        np.savetxt(dest_path, arr, fmt = '%i %s %s %s %s')
        
def remove_side(file_path):
    arr = np.loadtxt(file_path)
    if arr.shape[0] ==0:
        return np.array([])
    if len(arr.shape) == 1:
        if (0.92 <= arr[1] <= 0.96) and (0.67 <= arr[2] <= 0.75):
            return np.array([])
        else:
            return arr
    new_arr = []
    for n in arr:
        if not((0.92 <= n[1] <= 0.96) and (0.67 <= n[2] <= 0.75)):
            new_arr.append(n) 

    print(np.array(new_arr))
    return np.array(new_arr)

def remove_exept_highest(file_path):
    max_height = -1
    max_arr = None
    arr = np.loadtxt(file_path)
    if arr.shape[0] ==0:
        return 
    if len(arr.shape) == 1:
        return
    for n in arr:
        if n[4] > max_height:
            max_height = n[4]
            max_arr = n
    #return  np.expand_dims(max_arr, axis=0)
    np.savetxt(file_path, np.expand_dims(max_arr, axis=0), fmt = '%i %s %s %s %s')

def remove_exept_highest_mutiple(path_, index, dup):
    for i in range(index +1 , index + dup):
        file_dest = str(i).zfill(10) + ".txt"
        dest_path = join(path_, file_dest)
        remove_exept_highest(dest_path)



def find_more_than_one_row(path_, index, dup):
    arr12 = []
    for i in range(index +1 , index + dup):
        try:
            file_dest = str(i).zfill(10) + ".txt"
            dest_path = join(path_, file_dest)
            arr = np.loadtxt(dest_path)
            if len(arr.shape) ==2 and arr.shape[1] > 1:
                print(file_dest)
                arr12.append(file_dest)
        except:
            pass
    print(arr12)
    
def add_one_row(file_path, row):
    arr = np.loadtxt(file_path)
    arr = np.vstack([arr, row])
    return arr

def add_one_row_mutiple(path_, index, dup):
    for i in range(index +1 , index + dup):
        file_dest = str(i).zfill(10) + ".txt"
        dest_path = join(path_, file_dest)
        row = [0, 0.076172, 0.270833, 0.146094, 0.538889]
        arr = add_one_row(dest_path, row)
        np.savetxt(dest_path, arr, fmt='%i %s %s %s %s')

#_path = "/media/indoordesk/653ce34c-0c14-4427-8029-be7afe6d1989/video/front-25-03-17.41"
#add_one_row_mutiple(_path, 624, 10)