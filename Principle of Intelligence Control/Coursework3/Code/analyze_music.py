import os
import json
import librosa
import random

def analyze_music_and_save(audio_path, output_path):
    """
    分析音频的节拍并生成音符信息，保存到文件中。
    :param audio_path: 音频文件路径
    :param output_path: 保存音符数据的文件路径（JSON 格式）
    """
    print(f"正在分析音频文件：{audio_path}...")
    y, sr = librosa.load(audio_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    notes = []
    for beat_time in beat_times:
        # 随机生成音符类型和相关属性
        y_pos = random.choice([180, 540])  # 随机生成音符位置
        note_type = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]
        duration = random.randint(100, 200) if note_type == 1 else 0
        notes.append({
            "time": beat_time,
            "x": 1280,  # 初始 x 坐标
            "y": y_pos,
            "type": note_type,
            "duration": duration
        })

    # 保存到 JSON 文件
    with open(output_path, 'w') as f:
        json.dump(notes, f, indent=4)

    print(f"音符数据已保存到：{output_path}")


if __name__ == "__main__":
    music_dir = "/Users/youngbean/Desktop/Game/music_files"
    output_dir = "/Users/youngbean/Desktop/Game/notes_data"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(music_dir):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            audio_path = os.path.join(music_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
            analyze_music_and_save(audio_path, output_path)