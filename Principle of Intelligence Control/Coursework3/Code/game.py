def run_rhythm_game():
    import cv2
    import mediapipe as mp
    import numpy as np
    import random
    import time

    # 初始化 Mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # 音符类
    class Note:
        def __init__(self, x, y, note_type, duration=0):
            self.x = x
            self.y = y
            self.note_type = note_type  # 0: 普通, 1: 长按, 2: 连续打击
            self.hit = False
            self.hit_time = None
            self.duration = duration  # 长按音符的持续长度（红色）
            self.remaining_shakes = random.randint(3, 5) if note_type == 2 else 0  # 蓝色音符的摇头次数
            self.size = 40 if note_type == 2 else 20  # 蓝色音符起始大小

    # 全局变量
    screen_width, screen_height = 1280, 720
    notes = []
    note_speed = 10  # 音符移动速度
    spawn_interval = 1.0  # 音符生成间隔
    last_spawn_time = time.time()
    pause_all = False

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # 初始化变量
    previous_position = None
    previous_time = time.time()
    head_shake_speed = 0
    is_mouth_open = False

    # 响应线的位置
    response_line_x = 150

    # 音符生成函数
    def spawn_note():
        y = random.choice([screen_height // 4, screen_height // 4 * 3])  # 随机选择音符生成在上半或下半
        x = screen_width + 50  # 从右侧生成
        note_type = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]  # 随机生成普通、长按或连续打击音符
        duration = random.randint(100, 200) if note_type == 1 else 0  # 长按音符的持续长度
        notes.append(Note(x, y, note_type, duration))

    # 计算嘴部开合程度
    def calculate_mouth_open(face_landmarks):
        top_lip = face_landmarks.landmark[13].y
        bottom_lip = face_landmarks.landmark[14].y
        return bottom_lip - top_lip > 0.03  # 简单阈值判断嘴部是否张开

    # 游戏主循环
    while True:
        success, frame = cap.read()
        if not success:
            print("无法读取摄像头帧")
            break

        # 翻转帧以提供镜像效果
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # 获取当前时间
        current_time = time.time()

        # 状态信息初始化
        status_y = "N/A"
        status_mouth = "N/A"

        # 音符生成
        if not pause_all and current_time - last_spawn_time > spawn_interval:
            spawn_note()
            last_spawn_time = current_time

        # 更新音符位置
        for note in notes:
            if not note.hit and not pause_all:
                note.x -= note_speed

        # 绘制中间分隔线
        cv2.line(frame, (0, screen_height // 2), (screen_width, screen_height // 2), (255, 255, 255), 2)

        # 绘制响应线
        cv2.line(frame, (response_line_x, 0), (response_line_x, screen_height), (0, 255, 255), 2)

        # 绘制音符和爆裂效果
        for note in notes:
            if not note.hit:
                if note.note_type == 0:  # 绿色普通音符
                    cv2.circle(frame, (note.x, note.y), 20, (0, 255, 0), -1)
                elif note.note_type == 1:  # 红色长按音符（横线，变粗）
                    cv2.line(frame, (note.x, note.y), (note.x + note.duration, note.y), (0, 0, 255), 10)
                elif note.note_type == 2:  # 蓝色连续打击音符
                    cv2.circle(frame, (note.x, note.y), note.size, (255, 0, 0), -1)
                    cv2.putText(frame, str(note.remaining_shakes), (note.x - 10, note.y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif current_time - note.hit_time <= 0.3:  # 爆裂效果持续 0.3 秒
                for _ in range(5):
                    random_offset_x = random.randint(-10, 10)
                    random_offset_y = random.randint(-10, 10)
                    cv2.circle(frame, (note.x + random_offset_x, note.y + random_offset_y), 5, (255, 255, 255), -1)

        # 检测面部动作
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 获取鼻尖位置
                nose_tip = face_landmarks.landmark[1]
                head_position_y = nose_tip.y
                head_position_x = nose_tip.x

                # 计算水平移动速度
                if previous_position is not None:
                    delta_time = current_time - previous_time
                    head_shake_speed = abs(head_position_x - previous_position[0]) / delta_time
                else:
                    head_shake_speed = 0

                previous_position = (head_position_x, head_position_y)
                previous_time = current_time

                # 计算嘴部是否张开
                is_mouth_open = calculate_mouth_open(face_landmarks)
                # 更新状态信息
                status_y = "Upper Plane" if head_position_y < 0.5 else "Lower Plane"
                status_mouth = "Open" if is_mouth_open else "Closed"

                # 音符逻辑处理
                for note in notes:
                    if note.x == response_line_x:
                        if note.note_type == 0 and not note.hit:  # 绿色普通音符
                            if not is_mouth_open and ((note.y < screen_height // 2 and head_position_y < 0.5) or \
                            (note.y >= screen_height // 2 and head_position_y >= 0.5)):  # 检查闭嘴和平面条件
                                note.hit = True
                                note.hit_time = current_time

                        elif note.note_type == 1 and not note.hit:  # 红色长按音符
                            if note.x <= response_line_x <= note.x + note.duration:  # 红色音符头尾范围在响应线
                                if is_mouth_open and ((note.y < screen_height // 2 and head_position_y < 0.5) or \
                                                    (note.y >= screen_height // 2 and head_position_y >= 0.5)):
                                    note.duration -= note_speed  # 持续响应，逐渐减少线条长度
                                    note.x += note_speed  # 更新红色音符头部位置以保持响应线的头尾范围一致
                                    if note.duration <= 0:  # 音符完全响应
                                        note.hit = True
                                        note.hit_time = current_time
                            else:
                                note.in_response = False  # 一旦未满足条件则停止响应
                        elif note.note_type == 2 and not note.hit:  # 蓝色连续打击音符
                            pause_all = True  # 停止其他音符的移动
                            if not is_mouth_open and head_shake_speed > 0.2 and \
                            ((note.y < screen_height // 2 and head_position_y < 0.5) or \
                                (note.y >= screen_height // 2 and head_position_y >= 0.5)):  # 检查闭嘴和平面条件
                                note.remaining_shakes -= 1
                                note.size = max(10, note.size - 10)  # 音符变小
                                if note.remaining_shakes <= 0:  # 所有摇头完成
                                    note.hit = True
                                    note.hit_time = current_time
                                    pause_all = False




        # 绘制状态信息
        cv2.putText(frame, f"Head Position: {status_y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Mouth Status: {status_mouth}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Shake Speed: {head_shake_speed:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        # 显示画面
        cv2.imshow("Rhythm Game", frame)

        # 全屏显示
        cv2.namedWindow("Rhythm Game", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Rhythm Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # 按下 Esc 键退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

