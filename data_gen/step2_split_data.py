import random
from tqdm import tqdm
lines = open('geom-drugs/drugs.json').readlines()

random.shuffle(lines)
supply = (8- len(lines)%8)%8
lines +=lines[:supply]
train_files = []
for i in range(8):
    train_files.append(open(f'geom-drugs/train{i}.json','w'))
for i in tqdm(range(len(lines))):
    train_files[i%8].write(lines[i])
for i in range(8):
    train_files[i].flush()
    train_files[i].close()


    

