
### LAVAD
This method uses frame data as input. You need to firstly generate a ```annotations/``` file and a ```test.txt``` under it, which should be like:
```
0000_ball_anomaly_free 0 240 0
0001_ball_anomaly_free 0 240 0
0002_ball_anomaly_free 0 240 0
0003_ball_leak 0 240 0
...
```
where the first column is the name of the video file, and the third is the total frame number of the video. The second and the fourth are just 0.
The ```annotations``` file should at the same level of the ```frames``` file, that is:
```
frame_data/
├─ ball/
│  ├─ training/
│  │  ├─ frames/
│  │  ├─ annotations/
│  │  │  ├─ test.txt
```
After running the script you will find some new files like ```captions```, ```index``` etc. generated. They won't influence the original ```frames``` file.

### MemAE
This method uses ```frames``` as input. You should firstly generate a ```frames_idx``` folder by running the ```src/MemAE/matlab_script/matlabrunner.py```, the ```frames_idx``` folder should be at the same level as the ```frames``` folder.

### MGFN
This method uses ```i3d``` feature. Two lists of train or test video feature paths are required. Take test list for object ```ball``` for example, it should be like this:
```
path_to_your_data/ball/i3d/test/leak0000.npy
path_to_your_data/ball/i3d/test/leak0001.npy
path_to_your_data/ball/i3d/test/leak0002.npy
path_to_your_data/ball/i3d/test/leak0003.npy
path_to_your_data/ball/i3d/test/leak0005.npy
...
```
The final list should be a ```list``` file like ```test_ball.list```.

### VadCLIP
This method uses ```clip``` feature. This method also need two csvs like ```MGFN```, but we have automated this step. You only need to change the feature root path in ```src/VadCLIP/process.py```. If you want to modified to your own setting (e.g. the ratio of normal and abnormal instances) you can also look into ```src/VadCLIP/process.py``` to change them.

### S3R
This method uses ```i3d``` feature. The preparation for this method is relatively complex. You can find the preparation step in [Data preparation](../dataset/Readme.md), i3d feature part.