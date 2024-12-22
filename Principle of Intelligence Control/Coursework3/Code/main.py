import time
from real_time_emotion_recognition import main as real_time_main  # 假设 `real_time.py` 原始代码入口是 `main`
import argparse
from camera_test import test_main as camera_demo_main  # 假设 `camera_demo.py` 原始代码入口是 `main`
from game import run_rhythm_game

def real_time_emotion_recognition(duration):
    """运行表情识别模块，指定运行时长"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--haar', action='store_true', help='run the haar cascade face detector')
    parser.add_argument('--pretrained', type=str, default='checkpoint/model_weights/weights_epoch_75.pth.tar',
                        help='load weights')
    parser.add_argument('--head_pose', action='store_true', help='visualization of head pose euler angles')
    parser.add_argument('--path', type=str, default='', help='path to video to test')
    parser.add_argument('--image', action='store_true', help='specify if you test image or not')
    args = parser.parse_args()

    start_time = time.time()
    print("开始表情识别...")
    while time.time() - start_time < duration:
        real_time_main(args)  # 调用原始代码逻辑
        time.sleep(1)  # 控制运行频率

    print("表情识别完成，切换到手势识别阶段。")
    return True  # 返回标志表示表情识别阶段完成


def hand_gesture_recognition():
    return camera_demo_main()

# 状态变量
state = "emotion"  # 当前状态，可选值：emotion, gesture, game
emotion_duration = 10  # 表情识别运行时长（秒）
button_spawn_interval = 10  # 手势识别按钮生成间隔（秒）


def main():
    global state
    while True:
        if state == "emotion":
            print("进入表情识别阶段...")
            emotion_complete = real_time_emotion_recognition(emotion_duration)
            if emotion_complete:
                state = "gesture"  # 切换到手势识别阶段

        elif state == "gesture":
            print("进入手势识别阶段...")
            button_clicked = hand_gesture_recognition()
            if button_clicked:  # 如果按钮被手势点击，进入下一阶段
                state = "game"

        elif state == "game":
            print("进入节奏游戏阶段...")
            run_rhythm_game()
            print("节奏游戏完成，退出程序")
            break  # 游戏结束，退出循环


if __name__ == "__main__":
    main()