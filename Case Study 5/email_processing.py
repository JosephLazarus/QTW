#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 23:04:46 2021
@author: blanchard
"""

from os import listdir, getcwd, chdir
from os.path import isfile, join, dirname, realpath
import pandas as pd

def get_cwd():
    try:
        chdir(dirname(realpath(__file__)))
    except:
        chdir('/Users/bblanchard006/Desktop/SMU/QTW/Week 5/Summer 2021')

    active_dir = getcwd()
       
    return active_dir

def main():
    
    get_cwd()
    
    directories = [
            'sample_spam',
            'sample_ham'
        ]
    
    res_frame = pd.DataFrame()
        
    for d in directories:
        mypath = getcwd() + '/SpamAssassinMessages/' + d + '/'
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
        try:
            onlyfiles.remove('.DS_Store')
        except:
            pass
        
        for file in onlyfiles:
            with open(mypath + file, encoding='latin1') as f:
                lines = f.readlines()
                f.close()
                
            in_reply_count = 0
            sub_line_all_caps = 0
            attachments = 0
            subject_line = []
            n_lines = 0
            blank_lines = []
            
            for line in lines:
               n_lines += 1
               if "Subject: Re: " in line:
                   in_reply_count += 1
               if "Subject: " in line:
                   s_line = line.strip().replace('Subject: ','')
                   s_line = ''.join(e for e in s_line if e.isalnum())
                   num_upper = sum(1 for c in s_line if c.isupper())
                   ttl_chars = len(s_line)
                   if num_upper == ttl_chars:
                       sub_line_all_caps += 1
                   subject_line.append(s_line)
               if "content-type: multipart" in line.lower():
                   attachments += 1
               if line == "\n":
                   blank_lines.append(n_lines)

                   
        
            temp_frame = pd.DataFrame({
                        'filename':file,
                        'is_spam':['Y' if 'spam' in d else 'N'],
                        'in_reply': ['Y' if in_reply_count > 0 else 'N'], 
                        'subj_caps': ['Y' if sub_line_all_caps > 0 else 'N'], 
                        'attachments': ['Y' if attachments > 0 else 'N'], 
                        'body_lines': n_lines - min(blank_lines)
                        }, index=[0])
           
            res_frame = res_frame.append(temp_frame, ignore_index=True)
            
    res_frame.to_csv('output_file.csv', index=False)
    
    return res_frame

########################################
##### Main Function
########################################    

if __name__ == "__main__":
    res_frame = main()
    pass        
       


    
    
    
    