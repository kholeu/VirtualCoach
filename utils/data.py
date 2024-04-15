from torch.utils.data import Dataset
from torch import Tensor
from torchvision import transforms
import cv2

class VideoDataset(Dataset):
    """Torch dataset from a video file"""

    def __init__(self, video_stream, width, height):
        """Instance initialization

        Args:
            video_stream (cv2.VideoCapture): video stream
            width (int): frame width
            height (int): frame height
        """

        self.video = video_stream
        self.dim = (width, height)
        self.transform = transforms.ToTensor()

    def __len__(self):

        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, idx):
        """Get a frame from the video

        Args:
            idx (int): frame index

        Returns:
            frame (numpy array): frame (BGR colors) of shape (height, width, 3)
            transformed_frame (Tensor): transformed frame
        """

        self.video.set(cv2.CAP_PROP_POS_FRAMES, idx-1)
        res, frame = self.video.read()
        frame = cv2.resize(frame, self.dim, interpolation=cv2.INTER_AREA)
        return frame, self.transform(frame)
