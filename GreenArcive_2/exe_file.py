import sys
from openai import OpenAI
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage
from PyQt5 import uic, QtGui, QtCore
from temporal_model_by_webcam import capture_webcam, get_label

# UI파일 연결
form_class = uic.loadUiType("exe_file.ui")[0]


class WindowClass(QMainWindow, form_class):
    change_pixmap_signal = QtCore.pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # QLabel 객체 접근
        self.chat_label = self.findChild(QLabel, "chat")

        # QLabel 객체 접근
        self.emotion_label = self.findChild(QLabel, "emotion")

        # QQuickWidget 객체 접근 (video_widget)
        self.video_widget = self.findChild(QLabel, "video_widget")
        self.video_widget.setFixedSize(640, 480)  # 원하는 크기로 설정

        # QLineEdit 추가 (사용자 입력)
        self.user_input = self.findChild(QLineEdit, "user_input")
        self.user_input.setPlaceholderText("메시지를 입력하세요...")
        self.user_input.setEnabled(False)  # 초기에는 비활성화

        # QLineEdit에 returnPressed 시그널 연결
        self.user_input.returnPressed.connect(self.send_message)


        # QPushButton 추가 (메시지 전송 버튼)
        self.send_button = self.findChild(QPushButton, "send_button")
        self.send_button.setEnabled(False)  # 초기에는 비활성화
        self.send_button.clicked.connect(self.send_message)

        # QTextEdit 추가 (채팅 기록)
        self.chat_history = self.findChild(QTextEdit, "chat_history")
        self.chat_history.setReadOnly(True)

        # 채팅 시작 버튼 접근 (Qt Designer에서 만든 버튼)
        self.start_button = self.findChild(QPushButton, "start_button")
        self.start_button.clicked.connect(self.start_chat)

        # 시그널 연결
        self.change_pixmap_signal.connect(self.update_video_widget)

        # 비디오 캡처 시작
        self.start_video_capture()

        # 감정 인식 후 초기 메시지 설정
        self.last_known_emotion = None  # 마지막으로 알려진 감정
        self.conversation_history = []  # 이전 대화 기록

    def start_video_capture(self):
        self.video_thread = QtCore.QThread()
        self.worker = WebcamWorker(self.change_pixmap_signal)
        self.worker.moveToThread(self.video_thread)
        self.video_thread.started.connect(self.worker.run)
        self.video_thread.start()

    def update_video_widget(self, qt_image):
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        self.video_widget.setPixmap(pixmap)

    def start_chat(self):
        self.user_input.setEnabled(True)  # 사용자 입력 활성화
        self.send_button.setEnabled(True)  # 전송 버튼 활성화
        self.start_button.setEnabled(False)  # 채팅 시작 버튼 비활성화
        self.chat_history.append("AI: 채팅을 시작합니다. 기분에 대해 이야기해보세요!")

        # 초기 메시지 설정
        current_emotion = get_label() # 현재 감정 상태 가져오기
        print(f"Current Emotion: {current_emotion}")  # 감정 상태 출력
        emotion_message = self.get_initial_message(current_emotion)
        if emotion_message:
            self.chat_history.append(f"AI: {emotion_message}")

    def get_initial_message(self, emotion):
        # 감정 문자열에 따라 초기 메시지를 설정
        if emotion == 'Happiness':
            return "오늘 기분이 정말 좋으신가요? 기쁜 일이 있으셨나요?"
        elif emotion == 'Sadness':
            return "오늘 기분이 좀 우울하신가요? 어떤 일이 있었나요?"
        elif emotion == 'Anger':
            return "무언가 화가 나신 것 같아요. 어떤 일이 있었나요?"
        elif emotion == 'Surprise':
            return "무슨 일이 이렇게 놀라움을 주었나요?"
        elif emotion == 'Fear':
            return "무언가 걱정되는 일이 있으신가요?"
        elif emotion == 'Disgust':
            return "불쾌한 기분이 드시나요? 어떤 일이 있었나요?"
        else:  # Neutral
            return "오늘 기분이 어떤가요?"

    def send_message(self):
        user_message = self.user_input.text()
        if user_message:
            self.chat_history.append(f"나: {user_message}")
            self.conversation_history.append(user_message)  # 대화 기록 추가
            response = self.get_ai_response(user_message)
            self.chat_history.append(f"AI: {response}")
            self.user_input.clear()

    def get_ai_response(self, user_message):
        # 이전 대화 내용을 포함하여 AI에 요청
        if self.conversation_history:
            context = " ".join(self.conversation_history[-5:])  # 최근 5개 메시지
        else:
            context = ""

        client = OpenAI(api_key="YOUR_API_KEY")

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 심리상담가야."},
                {"role": "user", "content": context + " " + user_message}
            ]
        )
        response = completion.choices[0].message.content
        return response

class WebcamWorker(QtCore.QObject):
    def __init__(self, change_pixmap_signal):
        super().__init__()
        self.change_pixmap_signal = change_pixmap_signal

    def run(self):
        capture_webcam(self.change_pixmap_signal)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
