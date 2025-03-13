# -*- coding: utf- 8 -*-
"""
Created on Mon Aug 20 10:01:23 2018

@author: kirk
"""

# python routine for finding paths to specific EDR files. EDR files being
# pointed to are the auxiliaryy files

def find(name, path='/disk/kea/SDS/orig/supl/xtra-pds/'):
    import os
    for root, dirs, files in os.walk(path, followlinks = True):
        if name in files:
            return os.path.join(root, name)


with open('Western_Alba_Mons.txt','r') as content_file:
    content = content_file.read()
content_file.closed

fn = ''
with open('Western_Alba_Mons_Path.txt', 'w') as output_file:
    for line in content:
        if line != '\n':
            fn += line 
        else:
            fn += '_a.dat'
            a = find(fn)
            if a is not None: 
                print(a)
                b = a + '\n'
                output_file.write(b)
                del a, b
            else: 
                print('Warning',fn, 'not found')
            fn = ''
output_file.closed

        
