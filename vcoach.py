""" Virtual coach estimating pose based on reference one
   
    usage: virtual_coach.py [-h] [-b BATCH_SIZE] ref_videofile test_videofile out_videofile
 
    positional arguments:
      ref_videofile         Reference video file
      test_videofile        Estimated video file
      out_videofile         Output video file
    
    options:
      -h, --help            show this help message and exit
      -b BATCH_SIZE, --batch_size BATCH_SIZE
                            Batch size (default: 4)
"""

from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn
import torch
import numpy as np
import cv2
import functools, argparse

from utils.video import open_video, write_video
from utils.data import VideoDataset
from utils.metrics import *

def estimate_pose(pred_ref, pred_test, score_fn=cosine_similarity):
    """Estimate pose based on reference one

    Args:
        pred_ref (dict): reference pose prediction
        pred_test (dict): test pose prediction
        score_fn (callable): scoring function

    Returns:
        float: calculated score value
    """

    # Keypoints and confidences of instances with the greatest confidence: [0]
    keypoints_ref = pred_ref['keypoints'][0][:, :2].cpu()
    keypoints_test = pred_test['keypoints'][0][:, :2].cpu()
    confs_ref = pred_ref['keypoints_scores'][0].cpu()
    confs_test = pred_test['keypoints_scores'][0].cpu()
    # Reduced set of keypoints for increasing sensivity (excluded eyes and ears)
    scored_indices = torch.tensor([0, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16])
    keypoints_ref = torch.index_select(keypoints_ref, 0, scored_indices)
    keypoints_test = torch.index_select(keypoints_test, 0, scored_indices)
    confs_ref = torch.index_select(confs_ref, 0, scored_indices)
    confs_test = torch.index_select(confs_test, 0, scored_indices)

    # X*A + b = Y  ->  X * A = Y
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:, :-1]

    # Negative keypoint scores to zero
    confs_ref[confs_ref < 0] = 0
    confs_test[confs_test < 0] = 0

    # With keypoint weights
    W = np.sqrt(np.diag(confs_ref * confs_test))
    X = W @ pad(keypoints_test)
    Y = W @ pad(keypoints_ref)

    # Least squares for X * A = Y
    # A is the affine transformation
    A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    A[np.abs(A) < 1e-10] = 0  # low values to zero

    # Best fit of the test to the coach
    transform = lambda x: unpad(pad(x) @ A)
    keypoints_transformed = transform(keypoints_test)

    mutual_confs = np.sqrt(confs_ref * confs_test)
    score = score_fn(
        keypoints_transformed, keypoints_ref.numpy(),
        mutual_confs.numpy()
    )

    return score

def draw_skeleton_per_person(img, limbs, all_keypoints, all_scores, confs,
                             keypoint_threshold=2, conf_threshold=0.9, coach=True):
    """Draw skeleton per person on the image

    Args:
        img: input image
        limbs: list of connections between keypoints
        all_keypoints: tensor of keypoints locations, in [x, y, v] format
        all_scores: tensor of scores for all keypoints, for each detected person
        confs: tensor of confidence scores for detected person
        keypoint_threshold: threshold for keypoint score
        conf_threshold: threshold for confidence score of detected person
        coach: flag defining skeleton color

    Returns:
        img_copy: image with drawn skeleton
    """

    # initialize a color
    color = (0, 255, 0) if coach else (0, 255, 255)
    # create a copy of the image
    img_copy = img.copy()
    # check if the keypoints are detected
    if len(confs) > 0:
      # pick a set of N color-ids from the spectrum
      colors = np.arange(1, 255, 255//len(all_keypoints)).tolist()[::-1]
      # iterate for every person detected
      for person_id in range(len(confs)):
          # check the confidence score of the detected person
          if confs[person_id] > conf_threshold:
            # grab the keypoint-locations for the detected person
            keypoints = all_keypoints[person_id, ...]

            # iterate for every limb
            for limb_id in range(len(limbs)):
              # pick the start-point of the limb
              limb_loc1 = keypoints[limbs[limb_id][0], :2].cpu().numpy().astype(np.int32)
              # pick the start-point of the limb
              limb_loc2 = keypoints[limbs[limb_id][1], :2].cpu().numpy().astype(np.int32)
              # consider limb-confidence score as the minimum keypoint score among the two keypoint scores
              limb_score = min(all_scores[person_id, limbs[limb_id][0]], all_scores[person_id, limbs[limb_id][1]])
              # check if limb-score is greater than threshold
              if limb_score > keypoint_threshold:
                # draw the line for the limb
                cv2.line(img_copy, tuple(limb_loc1), tuple(limb_loc2), color, 2)

    return img_copy

def draw_skeletons_with_scores(frame_ref, frame_test, pred_ref, pred_test,
                               score):
    """Draw skeletons with score on the image

    Args:
        frame_ref: reference frame
        frame_test: test frame
        pred_ref: reference pose prediction
        pred_test: test pose prediction
        score: score value

    Returns:
        np.array: image with drawn skeletons and score
    """

    limbs = [
        [2, 0], [2, 4], [1, 0], [1, 3], [6, 8], [8, 10], [5, 7], [7, 9],
        [12, 14], [14, 16], [11, 13], [13, 15], [6, 5], [12, 11], [6, 12],
        [5, 11]
    ]

    # Draw skeletons
    frame_r = draw_skeleton_per_person(
        frame_ref, limbs, pred_ref['keypoints'],
        pred_ref['keypoints_scores'], pred_ref['scores']
    )
    frame_t = draw_skeleton_per_person(
        frame_test, limbs, pred_test['keypoints'],
        pred_test['keypoints_scores'], pred_test['scores'], coach=False
    )

    # Overlay estimates on test frame
    score_scaled = int(score * 10)
    score_scaled = 0 if score_scaled < 0 else score_scaled
    frame_s = cv2.putText(
        frame_t, f"Score: {score_scaled}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0,255,255), 2, cv2.LINE_AA
    )

    return np.concatenate((frame_r, frame_s), axis=1)

def vcoach(video_ref, video_test, output, batch_size, score_fn=cosine_similarity):
    """Estimate trainee movements based on reference video

    Args:
        video_ref (str): reference video filename
        video_test (str): trainee video filename
        output (str): output video filename
        batch_size (int): batch size
        score_fn (callable): scoring function

    Returns:
        dictionary: resulting dictionary
    """
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames_out, scores, ma_scores = [], [], []
    sliding_window = 7 if batch_size > 8 else batch_size - 1

    # Load reference and test videos and their properties
    cap_ref, cap_ref_fps, cap_ref_w, cap_ref_h = open_video(video_ref)
    cap_test, cap_test_fps, cap_test_w, cap_test_h = open_video(video_test)

    # Calculate resized width to stack horizontally later (adjusting height)
    scale = cap_ref_h / cap_test_h
    width = int(cap_test_w * scale)

    # Data loaders and model
    ds_ref = VideoDataset(cap_ref, cap_ref_w, cap_ref_h)
    dl_ref = DataLoader(ds_ref, batch_size=batch_size)
    ds_test = VideoDataset(cap_test, width, cap_ref_h)
    dl_test = DataLoader(ds_test, batch_size=batch_size)
    model = keypointrcnn_resnet50_fpn(weights='DEFAULT').eval().to(device)

    # Process data
    start_time = time()
    for (frames_r, batch_r), (frames_t, batch_t) in tqdm(zip(dl_ref, dl_test)):
        with torch.no_grad():
            preds_r = model(batch_r.to(device))
            preds_t = model(batch_t.to(device))

        # Model predictions to scores
        scores_batch = list(
            map(functools.partial(estimate_pose, score_fn=score_fn),
                preds_r, preds_t)
        )
        scores.extend(scores_batch)

        # Apply moving average to scores
        ma_scores.extend(
            moving_average(scores, sliding_window, len(scores_batch))
        )

        # Draw skeletons and put scores
        frames_batch = map(
            draw_skeletons_with_scores, frames_r.numpy(), frames_t.numpy(),
            preds_r, preds_t, ma_scores[-batch_size:]
        )
        frames_out.extend(list(frames_batch))

    duration = time() - start_time
    cap_ref.release()
    cap_test.release()
    write_video(frames_out, output, cap_ref_fps, (cap_ref_w + width, cap_ref_h))

    # GPU memory cleanup
    torch.cuda.empty_cache()

    result = {
        'avg_score': np.mean(scores),
        'score_fn': score_fn.__name__,
        'fps': len(frames_out) / duration,
        'duration': duration
    }

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python3 vcoach.py',
        description='This program estimates human movements in test video \
            compare to the supplied reference'
    )
    parser.add_argument('ref_videofile', type=str, help='reference video file')
    parser.add_argument('test_videofile', type=str, help='estimated video file')
    parser.add_argument('out_videofile', type=str, help='output video file')
    parser.add_argument(
        '-b', '--batch_size', type=int, default=4, help='batch size (default: 4)'
    )
    parser.add_argument(
        '--score_fn',
        choices=['cosine_similarity', 'inv_weighted_distance', 'product'],
        default='product',
        help='scoring function (default: product)'
    )
    args = parser.parse_args()

    match args.score_fn:
        case 'cosine_similarity':
            score_fn = cosine_similarity
        case 'inv_weighted_distance':
            score_fn = inv_weighted_distance
        case 'product':
            score_fn = product

    res = vcoach(
        args.ref_videofile,
        args.test_videofile,
        args.out_videofile,
        args.batch_size,
        score_fn
    )

    print(f"Scoring function: {res['score_fn']}")
    print(f"Average score: {res['avg_score'] * 10:.1f}")
    print(f"Processing time: {res['duration']:.0f} seconds")
    print(f"Processing speed: {res['fps']:.1f} fps")
