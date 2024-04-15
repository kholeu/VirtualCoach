import cv2


def open_video(fname):
    """Open a video file

    Args:
        fname (str or PosixPath): input video file

    Returns:
        video_capture: VideoCapture object
        fps: frames per second
        width: video width
        height: video height
    """

    try:
        video_capture = cv2.VideoCapture(fname)
    except cv2.error as error:
        print(f"[Error]: {error}")
        raise

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return video_capture, fps, width, height

def write_video(frames, fname, fps, size: tuple):
    """Write a video file from frames

    Args:
        frames (list of numpy array): list of frames (BGR colors)
        fname (str or PosixPath): output video file
        fps (float): frames per second
        size (tuple): output size [width, height]
    """

    video_out = cv2.VideoWriter(
        str(fname), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in frames:
        video_out.write(frame)
    video_out.release()
