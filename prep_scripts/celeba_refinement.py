import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import mtcnn
import face_detection

BASE_DIR = ''

ATTR_PATH = ''
ATTR_CSV_PATH = ''
POSE_PATH = ''
POSE_CSV_PATH = ''
PARTIAL_MASKS_DIR = ''
FULL_MASKS_DIR = ''

BASE_REFINED_IMG = ''
BASE_REFINED128_IMG = ''


def convert_attr_to_csv():
    with open(ATTR_PATH, 'r') as mfile:
        num_elems = int(mfile.readline())
        labels = (mfile.readline().strip()).split(' ')
        labels.insert(0,'image_id')


        rows = []
        for i in range(num_elems):
            line = (mfile.readline().strip()).split(' ')
            img_id = line[0]
            values = line[2:]
            values = [int(n) for n in values]

            row = [img_id] + values
            rows.append(row)

        df = pd.DataFrame(rows, columns=labels)        
    df.to_csv(ATTR_CSV_PATH, index=False)


def convert_pose_to_csv():
    with open(POSE_PATH, 'r') as mfile:
        num_elems = int(mfile.readline())
        labels = (mfile.readline().strip()).split(' ')
        labels.insert(0,'image_id')


        rows = []
        for i in range(num_elems):
            line = (mfile.readline().strip()).split(' ')
            img_id = line[0]
            values = line[1:]
            values = [float(n) for n in values]

            row = [img_id] + values
            rows.append(row)

        df = pd.DataFrame(rows, columns=labels)        
    df.to_csv(POSE_CSV_PATH, index=False)


def sum_masks():
    label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
    
    for i in range(30000):
        image_name = str(i)+'.jpg'
        folder_number = i // 2000
        img_base = np.zeros((512, 512))
        for idx, label in enumerate(label_list):
            filename = os.path.join(PARTIAL_MASKS_DIR, str(folder_number), str(i).rjust(5, '0') + '_' + label + '.png')
            if (os.path.exists(filename)):
                img = cv2.imread(filename)
                img = img[:, :, 0]
                img_base[img != 0] = 255
        
        cv2.imwrite(os.path.join(FULL_MASKS_DIR, image_name), img_base,  [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print(i)


def apply_mask():

    for i in range(30000):
        image_name = str(i)+'.jpg'
        img = cv2.imread(BASE_DIR + image_name)
        mask = cv2.imread(FULL_MASKS_DIR + image_name, cv2.IMREAD_GRAYSCALE)
        
        img = cv2.resize(img, (512,512), interpolation=cv2.INTER_NEAREST)

        inv_mask = mask.copy()

        a = mask>127
        mask[a] = True
        mask[~a] = False
        inv_mask[~a] = True
        inv_mask[a] = False
        

        mask3c = np.stack((mask,mask,mask), axis=2)
        inv_mask3c = np.stack((inv_mask, inv_mask, inv_mask), axis=2)

        bg = np.ones_like(img) * 255
        bg = bg * inv_mask3c
        img = img * mask3c
        img = img + bg

        # img = cv2.resize(img, (128,128), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(BASE_REFINED_IMG, image_name), img,  [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print(i)


def delete_bad_attrs():
    attrs = pd.read_csv(ATTR_CSV_PATH)
    pose = pd.read_csv(POSE_CSV_PATH)

    deletes = []
    for idx, row in attrs.iterrows():
        is_old = row['Young'] == -1
        has_hat = row['Wearing_Hat'] == 1
        has_glass = row['Eyeglasses'] == 1
        is_blurry = row['Blurry'] == 1

        if is_old or has_hat or has_glass or is_blurry:
            deletes.append(idx)
            os.remove(BASE_REFINED_IMG + row['image_id'])

        print(idx)
    attrs = attrs.drop(deletes)
    pose = pose.drop(deletes)
    attrs.to_csv(ATTR_CSV_PATH, index=False)
    pose.to_csv(POSE_CSV_PATH, index=False)




def delete_bad_pose():
    attrs = pd.read_csv(ATTR_CSV_PATH)
    pose = pd.read_csv(POSE_CSV_PATH)    

    deletes = []
    for idx, row in pose.iterrows():
        
        yaw, pitch, raw = row['Yaw'], row['Pitch'], row['Raw']

        if abs(yaw) > 12:
            deletes.append(idx)
            os.remove(BASE_REFINED_IMG + row['image_id'])
            continue
        if abs(pitch) > 16:
            deletes.append(idx)
            os.remove(BASE_REFINED_IMG + row['image_id'])
            continue
        if abs(raw) > 4:
            deletes.append(idx)
            os.remove(BASE_REFINED_IMG + row['image_id'])
            continue
    print(len(deletes))

            
    attrs = attrs.drop(deletes)
    pose = pose.drop(deletes)
    attrs.to_csv(ATTR_CSV_PATH, index=False)
    pose.to_csv(POSE_CSV_PATH, index=False)



def selective_refinement(names):
    names = [str(name)+'.jpg' for name in names]
    attrs = pd.read_csv(ATTR_CSV_PATH)
    pose = pd.read_csv(POSE_CSV_PATH)  

    deletes = []
    for idx, row in pose.iterrows():
        if row['image_id'] in names:
            deletes.append(idx)

    for name in names:
        os.remove(BASE_REFINED_IMG + name)

    attrs = attrs.drop(deletes)
    pose = pose.drop(deletes)
    attrs.to_csv(ATTR_CSV_PATH, index=False)
    pose.to_csv(POSE_CSV_PATH, index=False)

def resize():
    attrs = pd.read_csv(ATTR_CSV_PATH)
    pose = pd.read_csv(POSE_CSV_PATH)
    for idx, row in pose.iterrows():
        image_name = row['image_id']

        img = cv2.imread(BASE_REFINED_IMG + image_name)
        img = cv2.resize(img, (128,128))

        cv2.imwrite(os.path.join(BASE_REFINED128_IMG, image_name), img,  [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    



def detect_landmarks():
    RetinaFace = face_detection.build_detector("RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)
    attrs = pd.read_csv(ATTR_CSV_PATH)
    pose = pd.read_csv(POSE_CSV_PATH)

    landmarks = np.empty((attrs.shape[0], 10), dtype=np.int16)
    deletes = []
    for idx, row in pose.iterrows():
        image_name = row['image_id']

        img = cv2.imread(BASE_REFINED128_IMG + image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)

        _, lands = RetinaFace.batched_detect_with_landmarks(img)   

        lands = np.asarray(lands)
        lands = np.rint(lands)
        lands = lands.astype(int)

        if lands.shape != (1,1,5,2):
            print(image_name, ' REMOVED')
            os.remove(BASE_REFINED128_IMG + image_name)
            deletes.append(idx)
            continue

        lands = lands.flatten()
        landmarks[idx] = lands
    

    print(len(deletes))

    print('maxxxxx : ', np.max(landmarks), '    minnnnnnnnnnnnnnnnnnn : ', np.min(landmarks))

    landmarks = np.delete(landmarks, deletes, axis=0)
    print('maxxxxx : ', np.max(landmarks), '    minnnnnnnnnnnnnnnnnnn : ', np.min(landmarks))

    np.save('celeba_landmarks.npy', landmarks)

    attrs = attrs.drop(deletes)
    pose = pose.drop(deletes)
    attrs.to_csv(ATTR_CSV_PATH, index=False)
    pose.to_csv(POSE_CSV_PATH, index=False)


def refine_names():
    attrs = pd.read_csv(ATTR_CSV_PATH)
    pose = pd.read_csv(POSE_CSV_PATH)

    for idx, row in attrs.iterrows():
        old_name = row['image_id']
        new_name = (str(idx)+'.jpg').rjust(9, '0')
        attrs.at[idx, 'image_id'] = new_name
        pose.at[idx, 'image_id'] = new_name
        os.rename(BASE_REFINED128_IMG + old_name, BASE_REFINED128_IMG + new_name)


    attrs.to_csv(ATTR_CSV_PATH, index=False)
    pose.to_csv(POSE_CSV_PATH, index=False)


def gender_labels():
    attrs = pd.read_csv(ATTR_CSV_PATH)

    genders = attrs['Male']
    genders = genders.values
    np.save('celeba_genders.npy', genders)
    
