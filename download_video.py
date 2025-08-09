# download_video.py
from pytube import YouTube
import sys

def download(url, out="input_video.mp4"):
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
    if stream is None:
        raise RuntimeError("No mp4 stream found.")
    stream.download(filename=out)
    print("Downloaded:", out)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_video.py <youtube_url>")
    else:
        download(sys.argv[1])
