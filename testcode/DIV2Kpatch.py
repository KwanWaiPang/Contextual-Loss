from PIL import Image
import os
import random
fin = '../../data/DIV2K800_sub'
fout = '../../data/DIV2K_rank_256'

isExists=os.path.exists(fout)
if not isExists:
	os.makedirs(fout) 
	print(fout+' is created')
else:
	print(fout+' exists')


HR_size = 256
count = 0
final = 32000
for file in os.listdir(fin):
    file_fullname = fin + '/' +file
    img = Image.open(file_fullname)
    a = [240-HR_size/2, 240-HR_size/2, 240+HR_size/2, 240+HR_size/2]
    # rnd_h = random.randint(0, 480-HR_size)
    # rnd_w = random.randint(0, 480-HR_size)
    # a = [rnd_w, rnd_h, rnd_w + HR_size, rnd_h + HR_size]

    box = (a)
    roi = img.crop(box)
    out_path = fout + '/' + file
    roi.save(out_path)
    count+=1
    if count==final:
    	break