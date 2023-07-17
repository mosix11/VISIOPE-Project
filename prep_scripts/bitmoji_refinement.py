import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import mtcnn
import face_detection


BASE_DIR = ''
BASE_DIR_MAN = BASE_DIR + 'MAN/'
BASE_DIR_WOMAN = BASE_DIR + 'WOMAN/'
GENDER_NUMPY = ''
GENDERS_CSV = ''

BITMOJI_DATASET_BASE_DIR = ''
BITMOJI_DATASET_LANDMARKS_PATH = ''
BITMOJI_DATASET_GENDERS_PATH = ''

GAN_BITMOJI_BASE_DIR = ''
GAN_BITMOJI_LANDMARKS_PATH = ''
GAN_BITMOJI_LANDMARKS_CSV = ''



def get_sorted_file_names():
    paths = os.listdir(BASE_DIR)
    paths = [int(p.replace('.jpg', '')) for p in paths]
    paths = sorted(paths)
    paths = [str(p)+'.jpg' for p in paths]
    for i, img_name in enumerate(paths):
        if len(img_name) < 9:
            difference = 9 - len(img_name)
            img_name = difference * '0' + img_name
        paths[i] = img_name

    return paths


def merge():
    mans_path = os.listdir(BASE_DIR_MAN)
    womans_path = os.listdir(BASE_DIR_WOMAN)
    mans_path = [os.path.join(BASE_DIR_MAN, pth) for pth in mans_path]
    womans_path = [os.path.join(BASE_DIR_WOMAN, pth) for pth in womans_path]

    labels = ([1] * len(mans_path)) + ([-1] * len(womans_path))
    labels = np.asarray(labels)
    np.random.shuffle(labels)
    names = []

    for i in range(len(labels)):
        names.append((str(i)+'.jpg').rjust(8, '0'))
        newname = os.path.join(BASE_DIR, (str(i)+'.jpg').rjust(8, '0'))
        if labels[i] == 1:
            oldname = mans_path.pop()
        else :
            oldname = womans_path.pop()

        img = cv2.imread(oldname)
        
        cv2.imwrite(newname, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


    np.save('bitmoji_genders.npy', labels)
    names = np.asarray(names)
    merged = np.stack((names.T, labels.T), axis=1)
    df = pd.DataFrame(merged, columns=['image_id', 'is_male'])
    df.to_csv('genders.csv', index=False)






def crop_for_publish():
    RetinaFace = face_detection.build_detector("RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)
    genders_csv = pd.read_csv(GENDERS_CSV)


    for idx, row in genders_csv.iterrows():
        path = os.path.join(BASE_DIR, row['image_id'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detection = RetinaFace.detect(img)
        img_w = img.shape[1]
        img_h = img.shape[0]

        if detection.shape[0] == 0:
            print('Not found ', path)
            continue

        threshold = 100
        if row['image_id'] == '0281.jpg':
            threshold = 110

        for n, face_box in enumerate(detection):
            x1, y1, x2, y2, _ = face_box
            width = x2 - x1
            height = y2 - y1

            if width < threshold:
                continue

            
            x1 = int(x1 - width / 2)
            if x1 < 0 :
                x1 = 0

            y1 = int(y1 - height / 2)
            if y1 < 0 :
                y1 = 0

            x2 = int(x2 + width / 2)
            if x2 > img_w:
                x2 = img_w

            y2 = int(y2 + height / 3)
            if y2 > img_h:
                y2 = img_h


            new_width = x2 - x1
            new_height = y2 - y1
            
            cropped = img[y1:y2, x1:x2]

            max_dim = max(new_width, new_height)

            base = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 255

            if new_width >= new_height:
                base_y1 = int((max_dim - new_height) / 2)
                base[base_y1:base_y1+new_height,:] = cropped
            else:
                base_x1 = int((max_dim - new_width) / 2)
                base[:, base_x1:base_x1+new_width] = cropped

            base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
            base = cv2.resize(base, (384,384))

            cv2.imwrite(BITMOJI_DATASET_BASE_DIR + row['image_id'], base, [int(cv2.IMWRITE_JPEG_QUALITY), 100])



def lands_for_publish():
    RetinaFace = face_detection.build_detector("RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)
    mtcnn_model = mtcnn.MTCNN()
    attrs = pd.read_csv(BITMOJI_DATASET_GENDERS_PATH)

    landmarks = np.empty((attrs.shape[0], 10), dtype=np.int16)
    deletes = []
    for idx, row in attrs.iterrows():
        image_name = row['image_id']
        path = BITMOJI_DATASET_BASE_DIR + image_name

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)

        _, lands = RetinaFace.batched_detect_with_landmarks(img)   

        lands = np.asarray(lands)
        lands = np.rint(lands)
        lands = lands.astype(int)


        if lands.shape != (1,1,5,2):
            mtcnn_dets = mtcnn_model.detect_faces(img[0])
            if len(mtcnn_dets) == 0:
                print(path, ' REMOVED')
                # os.remove(BASE_DIR + path)
                deletes.append(idx)
                continue
            elif len(mtcnn_dets) > 1:
                print('More than 1 found for :', path)
            else:
                keypoints = mtcnn_dets[0]['keypoints']
                lands = np.zeros((5,2), np.int)
                lands[0] = keypoints['left_eye']
                lands[1] = keypoints['right_eye']
                lands[2] = keypoints['nose']
                lands[3] = keypoints['mouth_left']
                lands[4] = keypoints['mouth_right']

        lands = lands.flatten()
        landmarks[idx] = lands

    print(len(deletes))

    print('maxxxxx : ', np.max(landmarks), '    minnnnnnnnnnnnnnnnnnn : ', np.min(landmarks))
    landmarks = np.delete(landmarks, deletes, axis=0)
    print('maxxxxx : ', np.max(landmarks), '    minnnnnnnnnnnnnnnnnnn : ', np.min(landmarks))

    names = attrs['image_id'].values
    names = names[:,np.newaxis]
    data = np.concatenate((names, landmarks), axis=1)
    df = pd.DataFrame(data, columns=['image_id', 'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y', 'nose_x', 'nose_y', 'mouth_left_x', 'mouth_left_y', 'mouth_right_x', 'mouth_right_y'])
    df.to_csv(BITMOJI_DATASET_LANDMARKS_PATH, index=False)




def crop_for_gan():
    RetinaFace = face_detection.build_detector("RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)
    genders_csv = pd.read_csv(GENDERS_CSV)


    for idx, row in genders_csv.iterrows():
        path = os.path.join(BASE_DIR, row['image_id'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detection = RetinaFace.detect(img)
        img_w = img.shape[1]
        img_h = img.shape[0]

        if detection.shape[0] == 0:
            print('Not found ', path)
            continue

        threshold = 100
        if row['image_id'] == '0281.jpg':
            threshold = 110

        for n, face_box in enumerate(detection):
            x1, y1, x2, y2, _ = face_box
            width = x2 - x1
            height = y2 - y1

            if width < threshold:
                continue

            
            x1 = int(x1 - width / 3)
            if x1 < 0 :
                x1 = 0

            y1 = int(y1 - height / 3)
            if y1 < 0 :
                y1 = 0

            x2 = int(x2 + width / 3)
            if x2 > img_w:
                x2 = img_w

            y2 = int(y2 + height / 10)
            if y2 > img_h:
                y2 = img_h


            new_width = x2 - x1
            new_height = y2 - y1
            
            cropped = img[y1:y2, x1:x2]

            max_dim = max(new_width, new_height)

            base = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 255

            if new_width >= new_height:
                base_y1 = int((max_dim - new_height) / 2)
                base[base_y1:base_y1+new_height,:] = cropped
            else:
                base_x1 = int((max_dim - new_width) / 2)
                base[:, base_x1:base_x1+new_width] = cropped

            base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
            base = cv2.resize(base, (128,128))

            cv2.imwrite(GAN_BITMOJI_BASE_DIR + row['image_id'], base, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def lands_for_GAN():
    RetinaFace = face_detection.build_detector("RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)
    mtcnn_model = mtcnn.MTCNN()
    attrs = pd.read_csv(GENDERS_CSV)

    landmarks = np.empty((attrs.shape[0], 10), dtype=np.int16)
    deletes = []
    for idx, row in attrs.iterrows():
        image_name = row['image_id']
        path = GAN_BITMOJI_BASE_DIR + image_name

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)

        _, lands = RetinaFace.batched_detect_with_landmarks(img)   

        lands = np.asarray(lands)
        lands = np.rint(lands)
        lands = lands.astype(int)


        if lands.shape != (1,1,5,2):
            mtcnn_dets = mtcnn_model.detect_faces(img[0])
            if len(mtcnn_dets) == 0:
                print(path, ' REMOVED')
                # os.remove(BASE_DIR + path)
                deletes.append(idx)
                continue
            elif len(mtcnn_dets) > 1:
                print('More than 1 found for :', path)
            else:
                keypoints = mtcnn_dets[0]['keypoints']
                lands = np.zeros((5,2), np.int)
                lands[0] = keypoints['left_eye']
                lands[1] = keypoints['right_eye']
                lands[2] = keypoints['nose']
                lands[3] = keypoints['mouth_left']
                lands[4] = keypoints['mouth_right']

        lands = lands.flatten()
        landmarks[idx] = lands

    print(len(deletes))

    print('maxxxxx : ', np.max(landmarks), '    minnnnnnnnnnnnnnnnnnn : ', np.min(landmarks))
    # landmarks = np.delete(landmarks, deletes, axis=0)
    landmarks[deletes[0]] = [46, 62, 87, 63, 66, 84, 49, 94, 84, 93]

    print('maxxxxx : ', np.max(landmarks), '    minnnnnnnnnnnnnnnnnnn : ', np.min(landmarks))
    np.save(GAN_BITMOJI_LANDMARKS_PATH, landmarks)

    names = attrs['image_id'].values
    names = names[:,np.newaxis]
    data = np.concatenate((names, landmarks), axis=1)
    df = pd.DataFrame(data, columns=['image_id', 'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y', 'nose_x', 'nose_y', 'mouth_left_x', 'mouth_left_y', 'mouth_right_x', 'mouth_right_y'])
    df.to_csv(GAN_BITMOJI_LANDMARKS_CSV, index=False)


# img = cv2.imread('./MOSTAFA/BitmojiGAN/images/2208.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()

# lands_for_GAN()

# def normalize(img, current_min, current_max, desired_min, desired_max):
#     img = (img - current_min) * ((desired_max - desired_min) / (current_max - current_min)) + desired_min
#     return img

# def show_img(img, land):
#     img = normalize(img, 0, 255, 0, 1)
#     heatmap = np.zeros_like(img)
#     color = (255,255,255)
#     cv2.circle(heatmap, (land[0], land[1]), 2, color, 2)
#     cv2.circle(heatmap, (land[2], land[3]), 2, color, 2)
#     cv2.circle(heatmap, (land[4], land[5]), 2, color, 2)
#     cv2.circle(heatmap, (land[6], land[7]), 2, color, 2)
#     cv2.circle(heatmap, (land[8], land[9]), 2, color, 2)

#     mixed = np.clip(img + heatmap, 0, 1)

#     plt.imshow(mixed)
#     plt.show()

# lands = pd.read_csv(GAN_BITMOJI_LANDMARKS_CSV)
# landsnp = np.load(GAN_BITMOJI_LANDMARKS_PATH)

# for idx, row in lands.iterrows():
#     if idx % 50 == 0:
#         path = os.path.join(GAN_BITMOJI_BASE_DIR, row['image_id'])
#         # land = row[1:].values
#         land = landsnp[idx]
#         img = cv2.imread(path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         show_img(img, land)
            