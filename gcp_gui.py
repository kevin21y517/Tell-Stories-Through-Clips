import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from moviepy.editor import VideoFileClip
from PIL import Image, ImageTk
import time
from gcp_video import process_video
from ffpyplayer.player import MediaPlayer

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if file_path:
        progress_label.config(text="正在處理...")
        threading.Thread(target=process_video, args=(file_path, update_progress, process_complete)).start()

def update_progress(message):
    progress_label.config(text=message)

def process_complete(video_path):
    progress_label.config(text="處理完成")
    save_button.config(state=tk.NORMAL)
    global output_video_path
    output_video_path = video_path
    play_video(video_path)

def save_file():
    if output_video_path:
        save_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
        if save_path:
            with open(output_video_path, "rb") as src, open(save_path, "wb") as dst:
                dst.write(src.read())
            messagebox.showinfo("保存成功", f"影片已保存至 {save_path}")
        else:
            messagebox.showwarning("保存失敗", "未選擇保存路徑")
    else:
        messagebox.showwarning("保存失敗", "未處理完成的影片")

def play_video(video_path):
    def stream():
        video = VideoFileClip(video_path)
        player = MediaPlayer(video_path)
        start_time = time.time()
        canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()

        for frame in video.iter_frames(fps=video.fps, dtype="uint8"):
            frame_image = Image.fromarray(frame)
            frame_image = frame_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            frame_image = ImageTk.PhotoImage(frame_image)
            canvas.create_image(0, 0, anchor=tk.NW, image=frame_image)
            canvas.image = frame_image
            canvas.update()

            audio_frame, val = player.get_frame()
            if val != 'eof' and audio_frame is not None:
                img, t = audio_frame

            elapsed_time = time.time() - start_time
            expected_time = video.reader.pos / video.fps
            sleep_time = expected_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        video.close()
        player.close_player()

    threading.Thread(target=stream).start()

# 建立Tkinter應用程式
root = tk.Tk()
root.title("影片合成工具")
root.geometry("800x600")
root.configure(bg="#f0f0f0")

# 標題
title_label = tk.Label(root, text="影片合成工具", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
title_label.pack(pady=20)

# 選擇文件框
frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(pady=10)

label = tk.Label(frame, text="選擇一個MP4影片文件：", font=("Helvetica", 12), bg="#f0f0f0")
label.pack(side=tk.LEFT, padx=10)

button = tk.Button(frame, text="選擇文件", command=select_file, font=("Helvetica", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
button.pack(side=tk.LEFT, padx=10)

# 進度標籤
progress_label = tk.Label(root, text="", font=("Helvetica", 12), bg="#f0f0f0")
progress_label.pack(pady=10)

# 畫布框架
canvas_frame = tk.Frame(root, bg="#000000", bd=2, relief=tk.SUNKEN)
canvas_frame.pack(pady=10)
canvas = tk.Canvas(canvas_frame, width=640, height=480, bg="black")
canvas.pack()

# 保存按鈕
save_button = tk.Button(root, text="另存影片", command=save_file, state=tk.DISABLED, font=("Helvetica", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
save_button.place(relx=1.0, x=-20, y=20, anchor="ne")

output_video_path = None

root.mainloop()
