import os

anno_path = "/home/lc/Desktop/wza/gy/dyna/lavad-main/dataset/ball/annotations/test.txt"
vid_path = "/home/lc/Desktop/wza/gy/dyna/lavad-main/dataset/ball/frames"

frame_dict = {
 'fan': 180,
 'rolling_bearing': 60,
 'spherical_bearing': 60,
 'servo': 120,
 'clip': 120,
 'usb': 60,
 'hinge': 120,
 'screw': 180,
 'lock': 60,
 'gear': 120,
 'clock': 180,
 'slide': 120,
 'zipper': 120,
 'button': 180,
 'rubber band': 180,
 'liquid': 180,
 'caster_wheel': 180,
 'sticky_roller': 180,
 'magnet': 180,
 'toothpaste': 180,
 'ball':240
}

# with open(anno_path,"w") as f:
#     for vid in os.listdir(vid_path):
#         for key in frame_dict.keys():
#             if key in vid.split("_"):
#                 print(vid, 0, frame_dict[key], 0, file=f)

with open(anno_path,"w") as f:
    for vid in os.listdir(vid_path):
        frame_list = os.listdir(os.path.join(vid_path, vid))
        print(vid, 0, len(frame_list), 0, file=f)

# for vid in os.listdir(vid_path):
#     for key in frame_dict.keys():
#         if key in vid.split("_"):
#             frame_list = os.listdir(os.path.join(vid_path, vid))
#             frame_num = frame_dict[key]
#             if frame_num != len(frame_list):
#                 print(vid, len(frame_list), frame_num)
