from pathlib import Path
import shutil

for i in range(367, 370):
    Path('/home/shawn/CITS4402/content/data/volume_{}'.format(str(i))).mkdir(parents=True, exist_ok=True)
    for j in range(0, 155):
        shutil.move(
            '/home/shawn/CITS4402/content/data/volume_{}_slice_{}.h5'.format(str(i), str(j)), 
            '/home/shawn/CITS4402/content/data/volume_{}'.format(str(i)))

