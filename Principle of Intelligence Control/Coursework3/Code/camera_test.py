import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import time
import random


def generate_button_position(frame_width, frame_height):
    """生成按钮的随机位置"""
    button_width, button_height = 100, 50  # 固定按钮尺寸
    # 随机生成按钮的位置，确保按钮不会超出屏幕范围
    button_x = random.randint(0, frame_width - button_width)
    button_y = random.randint(0, frame_height - button_height)
    return (button_x, button_y, button_width, button_height)

def mouse_callback(event, x, y, flags, param):
    """鼠标点击回调函数"""
    global exit_game
    button_positions = param  # 获取最新的按钮位置

    if event == cv2.EVENT_LBUTTONDOWN:  # 检测鼠标左键点击
        for button_position in button_positions:
            button_x, button_y, button_width, button_height = button_position
            x2, y2 = button_x + button_width, button_y + button_height  # 计算右下角坐标

            # 输出调试信息
            print(f"Mouse click at ({x}, {y})")
            print(f"Button bounds: ({button_x}, {button_y}, {x2}, {y2})")

            if button_x <= x <= x2 and button_y <= y <= y2:
                print("Button clicked!")
                exit_game = True
                break
            else:
                print("Click outside the button.")

def create_fuzzy_controller():
    """
    创建一个模糊控制器，用于根据手部速度和位置误差来调整鼠标移动速度。
    """
    # 定义输入变量
    hand_speed = ctrl.Antecedent(np.arange(0, 21, 1), 'hand_speed')  # 速度范围 0-20
    position_error = ctrl.Antecedent(np.arange(0, 501, 1), 'position_error')  # 误差范围 0-500

    # 定义输出变量
    mouse_speed = ctrl.Consequent(np.arange(0, 11, 1), 'mouse_speed')  # 鼠标速度范围 0-10

    # 定义模糊集（隶属函数）
    hand_speed['slow'] = fuzz.trimf(hand_speed.universe, [0, 0, 10])
    hand_speed['medium'] = fuzz.trimf(hand_speed.universe, [5, 10, 15])
    hand_speed['fast'] = fuzz.trimf(hand_speed.universe, [10, 20, 20])

    position_error['low'] = fuzz.trimf(position_error.universe, [0, 0, 250])
    position_error['high'] = fuzz.trimf(position_error.universe, [150, 500, 500])

    # 为 mouse_speed 添加 'slow', 'medium', 'fast' 隶属函数
    mouse_speed['slow'] = fuzz.trimf(mouse_speed.universe, [0, 0, 5])
    mouse_speed['medium'] = fuzz.trimf(mouse_speed.universe, [3, 5, 7])  # 添加 'medium'
    mouse_speed['fast'] = fuzz.trimf(mouse_speed.universe, [5, 10, 10])

    # 定义规则
    rule1 = ctrl.Rule(hand_speed['slow'] & position_error['low'], mouse_speed['slow'])
    rule2 = ctrl.Rule(hand_speed['slow'] & position_error['high'], mouse_speed['slow'])
    rule3 = ctrl.Rule(hand_speed['medium'] & position_error['low'], mouse_speed['medium'])
    rule4 = ctrl.Rule(hand_speed['medium'] & position_error['high'], mouse_speed['fast'])
    rule5 = ctrl.Rule(hand_speed['fast'] & position_error['low'], mouse_speed['fast'])
    rule6 = ctrl.Rule(hand_speed['fast'] & position_error['high'], mouse_speed['fast'])

    # 创建控制系统
    mouse_speed_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
    mouse_speed_sim = ctrl.ControlSystemSimulation(mouse_speed_ctrl)

    return mouse_speed_sim

def detect_gesture(fingers):
    """
    根据手指状态检测当前手势。
    fingers: List[int]，包含四个元素，分别表示[食指, 中指, 无名指, 小指]是否伸直（1为伸直，0为弯曲）。
    返回: str 或 None，表示当前手势。
    """
    # 定义手势
    if fingers == [1, 0, 0, 0]:
        return 'move'
    elif fingers == [1, 1, 0, 0]:
        return 'click'
    elif fingers == [1, 1, 1, 1]:
        return 'right_click'
    else:
        return None

def test_main():
    global exit_game
    exit_game = False

    # 初始化 MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # 创建 Hand 模型
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1
    )
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 设置为更大的分辨率，适合全屏
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # 全屏设置
    cv2.namedWindow("Game", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 获取原始摄像头画面的尺寸
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 生成按钮的位置
    button_positions = []  # 用来存储所有按钮的位置
    last_button_time = time.time()  # 记录上次生成按钮的时间

    # 设置鼠标回调函数，传递button_positions
    cv2.setMouseCallback("Game", mouse_callback, param=button_positions)

    # 在第0秒生成第一个按钮
    button_position = generate_button_position(frame_width, frame_height)
    button_positions.append(button_position)
    last_button_time = time.time()  # 从此时开始计时

    previous_hand_position = None
    smoothing_factor = 0.2

    # 获取屏幕分辨率
    screen_width, screen_height = pyautogui.size()

    # 创建模糊控制器
    mouse_speed_sim = create_fuzzy_controller()

    # 初始化手势相关变量
    previous_gesture = None
    last_click_time = 0
    click_delay = 1.0  # 1 second delay to prevent multiple clicks

    while True:
        success, img = cap.read()
        if not success:
            print("无法读取摄像头帧")
            break

        # 镜像翻转（左右反转）以提供镜像效果
        img_flipped = cv2.flip(img, 1)

        # 转换为 RGB 格式
        img_rgb = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB)
        # 每10秒生成一个新按钮
        current_time = time.time()
        if current_time - last_button_time >= 10:
            button_position = generate_button_position(frame_width, frame_height)
            button_positions.append(button_position)
            last_button_time = current_time

        # 绘制所有按钮
        for button_position in button_positions:
            button_x, button_y, button_width, button_height = button_position
            cv2.rectangle(img_flipped, (button_x, button_y),
                          (button_x + button_width, button_y + button_height),
                          (0, 255, 0), -1)
            cv2.putText(img_flipped, "PLAY", (button_x + 20, button_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 处理图像并获取手部关键点
        results = hands.process(img_rgb)

        gesture = None

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 绘制手部关键点
                mp_draw.draw_landmarks(img_flipped, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 计算每个手指是否伸直（不考虑拇指）
                fingers = []

                # 获取手的类型（左手或右手）
                hand_label = handedness.classification[0].label  # 'Left' 或 'Right'

                # 食指
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                if index_tip.y < index_pip.y:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # 中指
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                if middle_tip.y < middle_pip.y:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # 无名指
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
                if ring_tip.y < ring_pip.y:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # 小指
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
                if pinky_tip.y < pinky_pip.y:
                    fingers.append(1)
                else:
                    fingers.append(0)

                num_fingers = fingers.count(1)

                # 识别手势
                gesture = detect_gesture(fingers)
                print(f"Detected gesture: {gesture}")

                # 设置模糊控制器的输入
                # 仅在 'move' 手势时调整鼠标移动速度
                if gesture == 'move':
                    # 计算手部速度和位置误差
                    if previous_hand_position is not None:
                        speed = np.linalg.norm([
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x - previous_hand_position[0],
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y - previous_hand_position[1]
                        ]) * 100  # 放大速度值以适应模糊控制器的输入范围
                        error = np.linalg.norm([
                            0.5 - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                            0.5 - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                        ]) * 1000  # 放大误差值以适应模糊控制器的输入范围
                    else:
                        speed = 0
                        error = np.linalg.norm([
                            0.5 - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                            0.5 - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                        ]) * 1000

                    previous_hand_position = [
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                    ]

                    mouse_speed_sim.input['hand_speed'] = speed
                    mouse_speed_sim.input['position_error'] = error

                    # 计算模糊输出
                    try:
                        mouse_speed_sim.compute()
                        calculated_mouse_speed = mouse_speed_sim.output['mouse_speed']
                        print(f"Calculated mouse speed: {calculated_mouse_speed}")
                    except Exception as e:
                        print(f"Fuzzy control computation error: {e}")
                        calculated_mouse_speed = 5  # 默认速度

                    # 映射手指位置到屏幕坐标
                    screen_x = np.interp(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, [0, 1], [0, screen_width])
                    screen_y = np.interp(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y, [0, 1], [0, screen_height])

                    # 使用平滑算法避免鼠标跳跃
                    current_mouse_x, current_mouse_y = pyautogui.position()
                    smooth_x = current_mouse_x + (screen_x - current_mouse_x) * smoothing_factor
                    smooth_y = current_mouse_y + (screen_y - current_mouse_y) * smoothing_factor

                    # 移动鼠标
                    pyautogui.moveTo(smooth_x, smooth_y, duration=0.01 * (11 - calculated_mouse_speed))
                else:
                    # 如果不是 'move' 手势，保持 previous_hand_position 不变
                    pass

                # 手势识别与执行鼠标操作
                if gesture in ['click', 'right_click']:
                    # 仅在手势状态改变且满足冷却时间时响应
                    current_time = time.time()
                    if gesture != previous_gesture and (current_time - last_click_time) > click_delay:
                        if gesture == 'click':
                            pyautogui.click()
                            print("Single Click Executed")
                        elif gesture == 'right_click':
                            pyautogui.rightClick()
                            print("Right Click Executed")
                        last_click_time = current_time
                        previous_gesture = gesture
                else:
                    # 如果当前手势是 'move' 或 'none'，重置 previous_gesture
                    if gesture != 'move':
                        previous_gesture = None

                # 显示手势
                if gesture:
                    cv2.putText(img_flipped, f'Gesture: {gesture}', (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    cv2.putText(img_flipped, f'Gesture: None', (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(img_flipped, f'FPS: {int(fps)}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示图像
        cv2.imshow("Game", img_flipped)
        # 检测退出条件
        if exit_game:
            print("游戏结束")
            break
        if cv2.waitKey(1) & 0xFF == 27:  # 按 Esc 键退出
            break

    cap.release()
    cv2.destroyAllWindows()
    return True
