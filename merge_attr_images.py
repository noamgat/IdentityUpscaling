import glob
import os
import shutil

os.chdir('paper/compare_attr')
os.makedirs('merged_attr_images', exist_ok=True)

for attr_num in [1, 20, 23, 31, 36]:
    for target_value in [0, 1]:
        for filename in glob.glob(f'adv_attr_{attr_num}_manual_{target_value}/*.jpg'):
            basename = os.path.splitext(os.path.basename(filename))[0]
            target_filename = f'merged_attr_images/{basename}_attr{attr_num}_val{target_value}.jpg'
            shutil.copy(filename, target_filename)