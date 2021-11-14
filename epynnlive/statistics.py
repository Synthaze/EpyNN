from epynn.commons.library import read_file
import glob
import sys
import re

lines_counter = {}

dir_list = [
'dummy_boolean/',
'dummy_image/',
'dummy_string/',
'dummy_time/',
'author_music/',
'captcha_mnist/',
'ptm_protein/'
]

dir_list = dir_list

print(len(dir_list), 'subdirectories')

desc = {}

print(dir_list)

desc['dummy_boolean/'] = ''
desc['dummy_image/'] = ''
desc['dummy_string/'] = ''
desc['dummy_time/'] = ''
desc['author_music/'] = ''
desc['captcha_mnist/'] = ''
desc['ptm_protein/'] = ''
desc['Total'] = ''



for dir in dir_list:

    if dir not in lines_counter.keys():
        lines_counter[dir] = {}

    for fname in [x for x in glob.glob(dir+'*py') if '__' not in x and 'ipynb' not in x]:

        if fname not in lines_counter[dir].keys():
            lines_counter[dir][fname] = {}
            lines_counter[dir][fname]['total'] = 0
            lines_counter[dir][fname]['docstring'] = 0
            lines_counter[dir][fname]['code'] = 0
            lines_counter[dir][fname]['inline'] = 0
            lines_counter[dir][fname]['block'] = 0
            lines_counter[dir][fname]['model'] = 0

        data = read_file(fname)
        data = '\n'.join([x for x in data.splitlines() if x.strip()])

        if 'train.py' in fname:
            for x in data.splitlines():
                if '# Model' in x:
                    lines_counter[dir][fname]['model'] += 1

        lines_counter[dir][fname]['total'] = len(data.splitlines())

        lines_counter[dir][fname]['block'] = len([x for x in data.splitlines() if '#' == x.strip()[0]])
        lines_counter[dir][fname]['inline'] = len([x for x in data.splitlines() if '#' in x and '#' != x.strip()[0]])

        lines_counter[dir][fname]['docstring'] = 0

        count = False
       

        for x in data.splitlines():

            if '"""' == x.strip()[0:3]:
                count = True

            elif '"""'  == x.strip()[0:3] and count:
                count = False
                lines_counter[dir][fname]['docstring'] += 1

            if count:
                lines_counter[dir][fname]['docstring'] += 1

            if '"""' == x.strip()[-3:]:
                count = False

        lines_counter[dir][fname]['code'] = lines_counter[dir][fname]['total'] - lines_counter[dir][fname]['docstring'] - lines_counter[dir][fname]['block']

        lines_counter[dir][fname]['comment_per_code'] = round((lines_counter[dir][fname]['inline'] + lines_counter[dir][fname]['block']) / lines_counter[dir][fname]['code'], 2)

directory_counter = {}

for dir in lines_counter.keys():
    directory_counter[dir] = {}
    directory_counter[dir]['files'] = len(lines_counter[dir].keys())
    directory_counter[dir]['total'] = 0
    directory_counter[dir]['docstring'] = 0
    directory_counter[dir]['code'] = 0
    directory_counter[dir]['inline'] = 0
    directory_counter[dir]['block'] = 0
    directory_counter[dir]['comment_per_code'] = 0
    directory_counter[dir]['model'] = 0


    for fname in lines_counter[dir].keys():
        for field in lines_counter[dir][fname].keys():
            directory_counter[dir][field] += lines_counter[dir][fname][field]
    

    directory_counter[dir]['comment_per_code'] = round((directory_counter[dir]['block'] + directory_counter[dir]['inline']) / directory_counter[dir]['code'], 2)

total_counter = {}

total_counter['files'] = 0
total_counter['total'] = 0
total_counter['docstring'] = 0
total_counter['code'] = 0
total_counter['inline'] = 0
total_counter['block'] = 0
total_counter['comment_per_code'] = 0
total_counter['model'] = 0

print('Subdirectory &', 'Files &', 'Lines &', 'Docstring &', 'Code &', 'Inline &', 'Block &', 'Comment &', 'Models &', 'Desc. & \\\\')
for dir in directory_counter.keys():
    print(dir.replace('/','').replace('_','\\_'), end=" & ")
    for field in directory_counter[dir].keys():
        print(directory_counter[dir][field], end=" & ")
        total_counter[field] += directory_counter[dir][field]
    print(desc[dir], end=" & ")
    print(' \\\\')



total_counter['comment_per_code'] = round((total_counter['block'] + total_counter['inline']) / total_counter['code'], 2)

print('Total &', end=" ")
for field in total_counter.keys():
    print(total_counter[field], end=" & ")

print(desc['Total'], end=" & ")
print('  \\\\')
