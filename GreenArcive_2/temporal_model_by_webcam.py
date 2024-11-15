import cv2
import mediapipe as mp #face detector
import math
import numpy as np
from PyQt5 import QtGui
import warnings
warnings.simplefilter("ignore", UserWarning)

# torch
import torch
from PIL import Image
from torchvision import transforms

# tf
import tensorflow as tf

# 파이토치를 위한 이미지 전처리 작업 이미지를 파이토치 모델에 입력하기 위해 전처리
def pth_processing(fp):
    class PreprocessInput(torch.nn.Module):
        def init(self):
            super(PreprocessInput, self).init()

        def forward(self, x):
            x = x.to(torch.float32)
            x = torch.flip(x, dims=(0,))
            x[0, :, :] -= 91.4953
            x[1, :, :] -= 103.8827
            x[2, :, :] -= 131.0912
            return x

    def get_img_torch(img):
        ttransform = transforms.Compose([
            transforms.PILToTensor(),
            PreprocessInput()
        ])
        img = img.resize((224, 224), Image.Resampling.NEAREST)
        img = ttransform(img)
        img = torch.unsqueeze(img, 0).to('cuda')
        return img

    return get_img_torch(fp) # fp 는 이미지 파일 경로를 말함

# 텐서플로를 위한 이미지 전처리 작업
def tf_processing(fp):
    def preprocess_input(x):
        x_temp = np.copy(x)
        x_temp = x_temp[..., ::-1]
        x_temp[..., 0] -= 91.4953
        x_temp[..., 1] -= 103.8827
        x_temp[..., 2] -= 131.0912
        return x_temp

    def get_img_tf(img):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
        img = tf.keras.utils.img_to_array(img)
        img = preprocess_input(img)
        img = np.array([img])
        return img

    return get_img_tf(fp)

# 정규화된 좌표를 실제 이미지 픽셀 좌표로 변환
def norm_coordinates(normalized_x, normalized_y, image_width, image_height):
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    # normalized_x, normarlized_y (정규화된 값들)
    # 원본 image_width, image_height (이미지 넓이 높이)
    return x_px, y_px

# 얼굴 랜드마크 좌표를 기반으로 얼굴의 경계 상자를 계산
# fl: 얼굴 랜드마크 객체
# w 이미지의 너비 h: 이미지의 높이
def get_box(fl, w, h):
    idx_to_coors = {}
    for idx, landmark in enumerate(fl.landmark):
        landmark_px = norm_coordinates(landmark.x, landmark.y, w, h)

        if landmark_px:
            idx_to_coors[idx] = landmark_px

    x_min = np.min(np.asarray(list(idx_to_coors.values()))[:, 0])
    y_min = np.min(np.asarray(list(idx_to_coors.values()))[:, 1])
    endX = np.max(np.asarray(list(idx_to_coors.values()))[:, 0])
    endY = np.max(np.asarray(list(idx_to_coors.values()))[:, 1])

    (startX, startY) = (max(0, x_min), max(0, y_min))
    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

    return startX, startY, endX, endY

# 이미지에 감정 결과를 표시
# img: 원본이미지
# box: 얼굴의 경계 상자 좌표(startX, startY, endX, endY)
# label: 에측된 감정레이블
# color: 경계 상자의 색상((기본값은 (128, 128, 128)).
# txt_color: 텍스트의 색상
# line_width: 경계 상자의 두께 기본값은 2
def display_EMO_PRED(img, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), line_width=2, ):
    lw = line_width or max(round(sum(img.shape) / 2 * 0.003), 2)
    text2_color = (255, 0, 255)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, text2_color, thickness=lw, lineType=cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX

    tf = max(lw - 1, 1)
    text_fond = (0, 0, 0)
    text_width_2, text_height_2 = cv2.getTextSize(label, font, lw / 3, tf)
    text_width_2 = text_width_2[0] + round(((p2[0] - p1[0]) * 10) / 360)
    center_face = p1[0] + round((p2[0] - p1[0]) / 2)

    cv2.putText(img, label,
                (center_face - round(text_width_2 / 2), p1[1] - round(((p2[0] - p1[0]) * 20) / 360)), font,
                lw / 3, text_fond, thickness=tf, lineType=cv2.LINE_AA)
    cv2.putText(img, label,
                (center_face - round(text_width_2 / 2), p1[1] - round(((p2[0] - p1[0]) * 20) / 360)), font,
                lw / 3, text2_color, thickness=tf, lineType=cv2.LINE_AA)
    return img

# 이미지 FPS 표시
# img 원본이미지
# text fps를 나타내는 텍스트
# margin 텍스트와 경계 상자 간의 여백 (기본 값1)
# box_scale 경계상자 크기 비율(기본값 1)
def display_FPS(img, text, margin=1.0, box_scale=1.0):
    img_h, img_w, _ = img.shape
    line_width = int(min(img_h, img_w) * 0.001)  # line width
    thickness = max(int(line_width / 3), 1)  # font thickness

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 0, 0)
    font_scale = thickness / 1.5

    t_w, t_h = cv2.getTextSize(text, font_face, font_scale, None)[0]

    margin_n = int(t_h * margin)
    sub_img = img[0 + margin_n: 0 + margin_n + t_h + int(2 * t_h * box_scale),
              img_w - t_w - margin_n - int(2 * t_h * box_scale): img_w - margin_n]

    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

    img[0 + margin_n: 0 + margin_n + t_h + int(2 * t_h * box_scale),
    img_w - t_w - margin_n - int(2 * t_h * box_scale):img_w - margin_n] = cv2.addWeighted(sub_img, 0.5, white_rect, .5,
                                                                                          1.0)

    cv2.putText(img=img,
                text=text,
                org=(img_w - t_w - margin_n - int(2 * t_h * box_scale) // 2,
                     0 + margin_n + t_h + int(2 * t_h * box_scale) // 2),
                fontFace=font_face,
                fontScale=font_scale,
                color=font_color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
                bottomLeftOrigin=False)

    return img


# label 전달하는 함수
label = ""
def get_label():
    global label
    return label

def capture_webcam(change_pixmap_signal):
    global label

    # mp_face_mesh: 얼굴의 랜드마크(특징점)를 탐지하는 역할
    mp_face_mesh = mp.solutions.face_mesh

    # backbone 모델과 lstm 모델을 불러온다
    # 백본 모델은 얼굴 이미지를 특성(feature)으로 변환합니다.
    # LSTM 모델은 백본 모델이 추출한 특징을 기반으로 감정을 예측합니다.
    name_backbone_model = '0_66_49_wo_gl'
    name_LSTM_model = 'IEMOCAP'


    # 가중치 불러오는 곳 파일형식에 따라 다르다. (pth(torch), h5(tensorflow))
    # torch(backbone가중치 + lstm 가중치)
    pth_backbone_model = torch.jit.load('models_EmoAffectnet/torchscript_model_{0}.pth'.format(name_backbone_model)).to(
        'cuda')
    pth_backbone_model.eval()

    pth_LSTM_model = torch.jit.load('models_EmoAffectnet/{0}.pth'.format(name_LSTM_model)).to('cuda')
    pth_LSTM_model.eval()


    # 감정을 라벨로 할당
    DICT_EMO = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}

    # w, h, fps 정의
    cap = cv2.VideoCapture(0) # 오픈cv 웹캠 실행
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    lstm_features = []



    #  mp_face_mesh.FaceMesh 는 얼굴 탐지 모델을 실행한다. while 루프로 매 프레임단위로 얼굴 인식한다.
    # startX, startY, endX, endY 은 얼굴 영역을 의미한다.
    with mp_face_mesh.FaceMesh(
            max_num_faces=1, # 인식할 얼굴 최대 개수
            refine_landmarks=False, # landmark 더욱 정밀하게 추적하는 기능 비활성(성능향상위함)
            min_detection_confidence=0.5, # 얼굴 감지 시 최소 0.5 신뢰도 이상일 때만 인식
            min_tracking_confidence=0.5) as face_mesh: # 추적시 신뢰도 0.5 이상일 때만 인식
        while cap.isOpened(): # 웹캠이 열려있을 때만 실행

            success, frame = cap.read() # 웹캠에서 새 프레임 가져와서 success, frame 에 전달
            if frame is None: break

            frame_copy = frame.copy()
            frame_copy.flags.writeable = False # frame_copy를 쓰기 불가능한 상태로 설정하여 성능 향상 (성능이 빠른 읽기모드로 변환)
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB) # 색상을 BGR(Opencv)에서 RGB로 변환(Mediapipe는 RGB요구하기 떄문)
            results = face_mesh.process(frame_copy) # RGB로 변환된 frame_copy를 face_mesh(얼굴탐지모델)에 보낸다
            frame_copy.flags.writeable = True # frame_copy를 다시 쓰기 가능한 상태로 되돌림 이후 분석 결과 다시 frame에 표시하거나 시각화 가능

            if results.multi_face_landmarks:
                for fl in results.multi_face_landmarks:
                    # get_box 함수를 이용해서 얼굴 영역 잘라낸다.
                    startX, startY, endX, endY = get_box(fl, w, h)
                    cur_face = frame_copy[startY:endY, startX: endX]

                    # torch
                    # 잘라낸 얼굴영역을 pth_processing 함수를 이용해서 정규화한다.
                    cur_face = pth_processing(Image.fromarray(cur_face))
                    # relu 함수 통과 시킨다. 음수 값을 0으로 만든다. backbonemodel을 통해 추출한 특징을 featrues에 저장
                    features = torch.nn.functional.relu(
                        pth_backbone_model.extract_features(cur_face)).cpu().detach().numpy()


                    if len(lstm_features) == 0: # 데이터가 없을 때 backbonemodel에서 추출한 features 데이터를 채운다.
                        lstm_features = [features] * 10
                    else: # 오래전 데이터 제거하고 새로운 데이터 채운다.
                        lstm_features = lstm_features[1:] + [features]

                    # torch
                    # lstm모델로 입력하기 위해 시퀀스 텐서로 변환하는 과정
                    lstm_f = torch.from_numpy(np.vstack(lstm_features)) # 10개의 lstm_features 을 numpy 배열(pytorch tensor)로 변환
                    # unsqueeze 텐서 차원추가 [sequence_length, feature_size] 를  [batch_size, sequence_length, feature_size] 로 변환(배치 차원 추가)
                    lstm_f = torch.unsqueeze(lstm_f, 0).to('cuda')
                    output = pth_LSTM_model(lstm_f).cpu().detach().numpy()


                    # detach(): 텐서를 연산 그래프로부터 분리하여, 이후 역전파나 미분이 계산되지 않게 합니다.
                    # cpu(): cpu로 옮긴다.


                    cl = np.argmax(output) # 가장 높은 감정 인덱스 찾고
                    label = DICT_EMO[cl] # 인덱스를 감정 label로 변환
                    frame = display_EMO_PRED(frame, (startX, startY, endX, endY), label, line_width=3)



            # OpenCV 프레임을 PyQt의 QImage로 변환
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # frame_copy 와 동일
            h, w, ch = rgb_image.shape

            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

            change_pixmap_signal.emit(qt_image)

