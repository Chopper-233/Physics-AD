import torch.utils.data as data
import numpy as np
from utils.mgfn_utils import process_feat
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import options.MGFN.option as option
args=option.parse_args()

class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False, is_preprocessed=False):
        self.modality = args.modality
        self.is_normal = is_normal
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
        else:
            self.rgb_list_file = args.rgb_list
        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None
        self.is_preprocessed = args.preprocessed

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            # if args.obj == 'UCF':
            #     if self.is_normal:
            #         self.list = self.list[810:]#ucf 810; sht63; xd 9525
            #         print('normal list')
            #         print(self.list)
            #     else:
            #         self.list = self.list[:810]#ucf 810; sht 63; 9525
            #         print('abnormal list')
            #         print(self.list)
            # elif args.obj == 'XD':
            #     if self.is_normal:
            #         self.list = self.list[9525:]
            #         print('normal list')
            #         print(self.list)
            #     else:
            #         self.list = self.list[:9525]
            #         print('abnormal list')
            #         print(self.list)
            # else:
            norm_dict = {
                'ball':8, 
                'fan':12, 
                'rolling_bearing':4, 
                'spherical_bearing':4, 
                'servo':12, 
                'clip':4, 
                'usb':4, 
                'hinge':8, 
                'screw':8, 
                'lock':8, 
                'gear':16, 
                'clock':8, 
                'slide':12, 
                'zipper':8, 
                'button':12, 
                'rubber_band':4, 
                'liquid':8, 
                'caster_wheel':4, 
                'sticky_roller':8, 
                'magnet':8, 
                'toothpaste':4, 
                'car':12
            }
            if self.is_normal:
                self.list = self.list[norm_dict[args.obj]:]
                print('normal list')
                print(self.list)
            else:
                self.list = self.list[:norm_dict[args.obj]]
                print('abnormal list')
                print(self.list)


    def __getitem__(self, index):
        label = self.get_label(index)  # get video level label 0/1
        if "train" in self.list[index].strip('\n') or "anomaly_free" in self.list[index].strip('\n'):
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        # if args.obj == 'XD':
        #     features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        #     features = np.array(features, dtype=np.float32)
        #     name = self.list[index].split('/')[-1].strip('\n')[:-4]
        # else:
            # if args.obj == 'UCF':
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)
        name = self.list[index].split('/')[-1].strip('\n')[:-4]
        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            
            # if args.obj == 'XD':
            #     mag = np.linalg.norm(features, axis=1)[:, np.newaxis]
            #     features = np.concatenate((features, mag), axis=1)
            # else: 
                # args.obj == 'UCF':
            mag = np.linalg.norm(features, axis=2)[:,:, np.newaxis]
            features = np.concatenate((features,mag),axis = 2)
            return features, name, label
        else:
            # if args.obj == 'XD':
            #     feature = process_feat(features, 32)
            #     if args.add_mag_info == True:
            #         feature_mag = np.linalg.norm(feature, axis=1)[:, np.newaxis]
            #         feature = np.concatenate((feature,feature_mag),axis = 1)
            #     return feature, label
            # else:
                # if args.obj == 'UCF':
            if self.is_preprocessed:
                return features, label
            features = features.transpose(1, 0, 2)  # [10, T, F]
            divided_features = []

            divided_mag = []
            for feature in features:
                feature = process_feat(feature, args.seg_length) #ucf(32,2048)
                divided_features.append(feature)
                divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
            divided_features = np.array(divided_features, dtype=np.float32)
            divided_mag = np.array(divided_mag, dtype=np.float32)
            divided_features = np.concatenate((divided_features,divided_mag),axis = 2)
            return divided_features, label


    def get_label(self, index):
        if self.is_normal:
            # label[0] = 1
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
            # label[1] = 1
        return label

    def __len__(self):

        return len(self.list)


    def get_num_frames(self):
        return self.num_frame
