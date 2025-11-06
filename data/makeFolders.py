# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:33:21 2022

@author: Edwin
"""

import os, shutil
cwd = os.path.dirname(os.path.abspath(__file__))
baseDir = os.path.join(cwd,'..')

#%% First empty folder
# try:
#     shutil.rmtree('data')
# except:
#     print()   

muscles = ['GMs1', 'GMs2', 'GMs3']
h1 = 'data'

#%% Data folders
# h3 = 'dataExp'
# for h2 in muscles:
#     for h4 in ['QR','SR','ISOM']:
#         folder = os.path.join(baseDir,h1,h2,h3,h4)
#         os.makedirs(folder)

# h3 = 'simsExp'
# for h2 in muscles:
#     for h4 in ['TM', 'IM']:
#         for h5 in ['QR','SR','ISOM']:
#             folder = os.path.join(baseDir,h1,h2,h3,h4,h5)
#             os.makedirs(folder)

# h3 = 'dataMC'
# for h2 in muscles[0:3]:
#     for h4 in ['{:02d}'.format(idx) for idx in range(1,51)]:
#         for h5 in ['QR','SR','ISOM']:
#             folder = os.path.join(baseDir,h1,h2,h3,h4,h5) 
#             os.makedirs(folder)

h3 = 'simsMC'
for h2 in muscles[0:3]:
    for h4 in ['{:02d}'.format(idx) for idx in range(1,51)]:
        for h5 in ['QR','SR','ISOM']:
            folder = os.path.join(baseDir,h1,h2,h3,h4,h5) 
            os.makedirs(folder)

# h3 = 'parameters'
# for h2 in muscles:
#     folder = os.path.join(baseDir,h1,h2,h3) 
#     os.makedirs(folder)
#     if h2 == 'GMs1' or h2 == 'GMs2' or h2 == 'GMs3':
#         os.makedirs(folder+'/mc')
#         os.makedirs(folder+'/interdep')
        

