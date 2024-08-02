from moviepy.editor import VideoFileClip

def get_video_duration_ms(video_path):
    clip = VideoFileClip(video_path)
    duration_seconds = clip.duration
    duration_ms = duration_seconds * 1000
    clip.reader.close()
    clip.audio.reader.close_proc()
    return duration_ms
