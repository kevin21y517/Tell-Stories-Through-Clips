import requests
import base64
import json
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip
import pysrt
from pydub import AudioSegment
from pydub.silence import detect_silence
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import librosa
import torch
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor

# 檢查設備是否可用
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 創建pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    chunk_length_s=30,
    device=device,
)

# 設置GCP Vision API和Cloud Text-to-Speech API金鑰
GCP_API_KEY = "AIzaSyAAhj77CeoB7NnOhiWYj9x9Svht-vnvvdE"
GPT_API_KEY = "sk-YihjYqqBhjbhXIxOn8lvY6HF8ql9yfyNz2Ovdehzfo2ucu1m"

def analyze_image(content):
    """ 使用GCP Vision API分析圖片 """
    vision_api_url = f"https://vision.googleapis.com/v1/images:annotate?key={GCP_API_KEY}"
    headers = {"Content-Type": "application/json"}
    body = {
        "requests": [{
            "image": {"content": content},
            "features": [{"type": "LABEL_DETECTION"}]
        }]
    }

    response = requests.post(vision_api_url, headers=headers, json=body)
    response_data = response.json()

    if "error" in response_data:
        raise Exception(response_data["error"]["message"])

    labels = response_data["responses"][0].get("labelAnnotations", [])
    descriptions = [label["description"] for label in labels]

    return descriptions

def generate_story_gpt(descriptions, context_story, max_tokens):
    """ 使用GPT API生成故事 """
    content = " ".join(descriptions)
    gpt_api_url = "https://api.chatanywhere.cn/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GPT_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "你是個厲害的說書人，請直接開始說故事。"},
            {"role": "user", "content": f"這是一個連貫的故事的一部分。以下是前文內容：{context_story}\n\n請根據這些描述繼續講述故事：{content}"}
        ],
        "max_tokens": max_tokens,  # 確保 max_tokens 設置合理
        "n": 1,
        "temperature": 0.7
    }

    response = requests.post(gpt_api_url, headers=headers, json=body)
    response_data = response.json()

    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response_data.get('error', 'Unknown error')}")

    choices = response_data.get("choices", [])
    if not choices:
        raise Exception("No choices returned from the API")

    story = choices[0].get("message", {}).get("content", "").strip()
    return story

def split_story_to_limit_words(story, words_per_minute):
    """ 根據每分鐘字數限制分割故事 """
    words = story.split()
    segments = []
    segment = []

    for word in words:
        segment.append(word)
        if len(segment) >= words_per_minute:
            segments.append(" ".join(segment))
            segment = []

    if segment:
        segments.append(" ".join(segment))

    return segments

def text_to_speech(text, output_audio_path):
    """ 使用Cloud Text-to-Speech API將文字轉換成語音 """
    tts_api_url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GCP_API_KEY}"
    headers = {"Content-Type": "application/json"}
    body = {
        "input": {"text": text},
        "voice": {"languageCode": "zh-CN", "ssmlGender": "FEMALE"},
        "audioConfig": {"audioEncoding": "MP3"}
    }
    # cmn-TW  cmn-TW-Wavenet-A

    response = requests.post(tts_api_url, headers=headers, json=body)
    response_data = response.json()

    if "error" in response_data:
        raise Exception(response_data["error"]["message"])

    audio_content = base64.b64decode(response_data["audioContent"])

    with open(output_audio_path, "wb") as audio_file:
        audio_file.write(audio_content)

def find_silence_intervals(audio, min_silence_len=300, silence_thresh=-40):
    silence_intervals = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    return silence_intervals

def split_audio_on_punctuation(audio, silence_intervals, max_duration):
    start = 0
    best_end = 0

    for silence in silence_intervals:
        end = silence[1]
        if end - start > max_duration:
            break
        best_end = end

    # 如果沒有找到合適的間隔，則取max_duration
    if best_end == 0:
        best_end = max_duration

    segment = audio[start:best_end]
    return segment


def merge_audio_with_video(video_path, audio_path, output_path):
    """ 將音頻與視頻合併 """
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    start_time = 0
    audio_duration = audio.duration
    video_duration = video.duration
    max_audio_duration = video_duration * 1000

    if audio_duration > video_duration:
        audio_segment = AudioSegment.from_file(audio_path)
        silence_intervals = find_silence_intervals(audio_segment)
        audio_segment = split_audio_on_punctuation(audio_segment, silence_intervals, max_audio_duration)
        audio = AudioFileClip(audio_segment.export("GCP/data/story.mp3", format="mp3").name)
        print("音频已分割")

    video_with_audio = video.set_audio(audio.set_start(start_time))
    video_with_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")

    return start_time, audio_duration


def transcribe_audio_with_whisper(audio_segment):
    """ 使用Huggingface的pipeline進行音頻轉錄 """
    prediction = pipe(audio_segment, batch_size=8, return_timestamps=True)["chunks"]
    return prediction

def split_text_by_comma_with_timestamps(chunk):
    """ 根據逗號分割文本並保留時間戳 """
    segments = []
    sentences = chunk['text'].split(',')
    current_start = chunk['timestamp'][0]
    duration = (chunk['timestamp'][1] - chunk['timestamp'][0]) / len(sentences)  # 平均分配時間戳

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        segments.append({
            'text': sentence,
            'timestamp': (current_start, current_start + duration)
        })
        current_start += duration

    return segments

def create_srt_from_prediction(prediction, output_file):
    """ 根據轉錄結果生成SRT文件 """
    subs = pysrt.SubRipFile()
    for chunk in prediction:
        text_segments = split_text_by_comma_with_timestamps(chunk)
        for segment in text_segments:
            start_seconds = segment['timestamp'][0]
            end_seconds = segment['timestamp'][1]
            text = segment['text']

            start = pysrt.SubRipTime(seconds=int(start_seconds), milliseconds=int((start_seconds % 1) * 1000))
            end = pysrt.SubRipTime(seconds=int(end_seconds), milliseconds=int((end_seconds % 1) * 1000))

            sub = pysrt.SubRipItem(index=len(subs) + 1, start=start, end=end, text=text.strip())
            subs.append(sub)

    subs.save(output_file, encoding='utf-8')
    print("SRT 文件已保存")

def create_subtitle_image(text, size, video_size, font_path="C:/Windows/Fonts/mingliu.ttc"):
    base_font_size = 24  # 基礎字體大小
    # 根據視頻分辨率調整字體大小
    font_size = int(base_font_size * (video_size[1] / 480))
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
    position = ((size[0] - text_size[0]) // 2, size[1] - text_size[1] - 10)
    draw.text(position, text, font=font, fill="white")
    return np.array(img)

def create_subtitle_clip(text, start_time, duration, video_size):
    subtitle_img = create_subtitle_image(text, video_size, video_size)
    subtitle_clip = ImageClip(subtitle_img).set_duration(duration).set_start(start_time).set_position(('center', 'bottom'))
    return subtitle_clip

def add_subtitles_to_video(video_path, srt_path, output_path):
    video = VideoFileClip(video_path)
    subtitles = pysrt.open(srt_path)
    subtitle_clips = []

    for sub in subtitles:
        start_time = sub.start.ordinal / 1000
        end_time = sub.end.ordinal / 1000
        duration = end_time - start_time
        subtitle_clips.append(create_subtitle_clip(sub.text, start_time, duration, video.size))

    final_video = CompositeVideoClip([video] + subtitle_clips)
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

def save_story_to_file(descriptions, story, file_path="GCP/data/story.txt"):
    """ 保存描述和故事到文件 """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("描述:\n")
        for description in descriptions:
            f.write(f"- {description}\n")
        f.write("\n生成的故事:\n")
        f.write(story)

def analyze_video(video_path, progress_callback):
    """ 分析影片中的每5幀圖片，提取描述並每5秒生成一批描述 """
    video = cv2.VideoCapture(video_path)
    descriptions = []
    frame_count = 0
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    descriptions_batches = []

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % int(frame_rate * 5) == 0:
            success, encoded_image = cv2.imencode(".jpg", frame)
            if success:
                image_bytes = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
                frame_descriptions = analyze_image(image_bytes)
                descriptions.extend(frame_descriptions)

        if frame_count % int(frame_rate * 60) == 0 and descriptions:
            descriptions_batches.append(descriptions.copy())
            descriptions.clear()

        frame_count += 1

    if descriptions:
        descriptions_batches.append(descriptions)

    video.release()
    return descriptions_batches

def process_video(video_path, progress_callback, completion_callback):
    try:
        progress_callback("正在分析影片...")
        video = VideoFileClip(video_path)
        video_duration = video.duration
        descriptions_batches = analyze_video(video_path, progress_callback)

        story_segments = []
        max_tokens_per_batch = 200  # 設置每個批次的最大tokens數
        words_per_minute = 150  # 每分鐘字數上限
        context_story = ""  # 前文內容

        progress_callback("正在生成故事...")
        for i, descriptions in enumerate(descriptions_batches):
            # 将每个描述批次的内容和前一段的故事内容连接起来，以保持前后文的关联
            story_segment = generate_story_gpt(descriptions, context_story, max_tokens_per_batch)
            context_story += " " + story_segment
            segmented_story = split_story_to_limit_words(story_segment, words_per_minute)
            story_segments.extend(segmented_story)

        complete_story = " ".join(story_segments)
        temp_audio_path = "GCP/data/temp_story.mp3"
        text_to_speech(complete_story, temp_audio_path)
        save_story_to_file([item for sublist in descriptions_batches for item in sublist], complete_story)
        print(f"生成的故事已保存到 story.txt 中")

        # 生成語音
        progress_callback("正在生成語音...")
        audio_path = "GCP/data/story.mp3"
        output_video_path = "GCP/data/output_video.mp4"
        text_to_speech(complete_story, audio_path)

        # 合併音頻和視頻
        progress_callback("正在合併音頻和視頻...")
        start_time, audio_duration = merge_audio_with_video(video_path, audio_path, output_video_path)
        print(f"合成的影片已保存到 {output_video_path} 中")

        # 生成SRT文件
        progress_callback("正在生成字幕...")
        audio, sr = librosa.load(audio_path, sr=16000)
        prediction = transcribe_audio_with_whisper(audio)
        srt_output_file = "GCP/data/output.srt"
        create_srt_from_prediction(prediction, srt_output_file)
        print(f"SRT 文件已保存為 {srt_output_file}")

        # 添加字幕到影片
        progress_callback("正在添加字幕到影片...")
        final_output_path = "GCP/data/final_output_video.mp4"
        add_subtitles_to_video(output_video_path, srt_output_file, final_output_path)
        print(f"最終合成的影片已保存到 {final_output_path}")

        progress_callback("完成")
        completion_callback(final_output_path)

    except Exception as e:
        print(f"錯誤: {e}")
        progress_callback(f"錯誤: {e}")