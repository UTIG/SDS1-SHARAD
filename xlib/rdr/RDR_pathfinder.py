__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu']\
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'February 20 2019'
         'author': 'Kirk Scanlan UTIG'
         'info': 'Tool to find paths to specific files on the hierarchy'}}

# python routine for finding paths to specific EDR files. EDR files being
# pointed to are the auxiliaryy files

def find(name, path='/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/RDR/mrosh_2001/data/geom'):
    import os
    for root, dirs, files in os.walk(path, followlinks = True):
        if name in files:
            return os.path.join(root, name)


with open('Cyril_RDRReferenceArea.txt','r') as content_file:
    content = content_file.read()
content_file.closed

fn = ''
with open('Cyril_RDRReferenceArea_hier.txt', 'w') as output_file:
    for line in content:
        if line != '\n':
            fn += line 
        else:
            fn += '.tab'
            a = find(fn)
            if a is not None: 
                #print(a)
                b = a + '\n'
                output_file.write(b)
                del a, b
            else: 
                print('Warning',fn, 'not found')
            fn = ''
output_file.closed

        
