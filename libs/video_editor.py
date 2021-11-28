import ffmpeg
from libs.ffprobe import probe_key_frames
from pathlib import Path
import os
import glob


def extract_frames(video_path, frame_n_size):

    for filename in glob.glob(os.path.join(os.environ['FRAMES_DIR'], '*.jpg')):
        Path(filename).unlink()

    kwargs = {
        'pix_fmt': 'rgb24',
        "loglevel": "error",
        'q:v': '2',
    }
    if frame_n_size > 1:

        kwargs.update(
        {'vf': f"select=not(mod(n\,{frame_n_size}))",
        'vsync': 'vfr'}
        )
    (
        ffmpeg
        .input(video_path)
        .output(os.path.join(os.environ['FRAMES_DIR'],'%5d.jpg'), **kwargs)
        .run()
    )


def key_frames(video_path):

    kwargs = {
        'select_streams': 'v',
        'show_entries': 'frame=pict_type',
    }

    return probe_key_frames(video_path, **kwargs)
