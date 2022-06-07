# # copy dataset to current folder 
# echo 'Copying data to current folder'
# ## miniimagenet
# cp /mnt/vinai/projects/poodle/dataset/images.zip data
# ## tiered imagenet
# cp /mnt/vinai/projects/poodle/dataset/tiered-imagenet.tar data
# ## CUB
# cp /mnt/vinai/projects/poodle/dataset/CUB_200_2011.tgz data

# Unzip
echo 'Extract compressed files'
cd data && 
unzip images.zip && 
tar -xvf tiered-imagenet.tar && 
tar -xvf CUB_200_2011.tgz &&
cd ..

# remove 
echo 'Removing redundant files'
cd data && 
rm images.zip 
rm tiered-imagenet.tar
rm CUB_200_2011.tgz
cd ..

# Run split
echo 'Splitting dataset'
### tiered standard split
python src/utils/data_split/tieredImagenet.py
### cub standard split (a closer look - chen et al., 2019)
python src/utils/data_split/cub.py

