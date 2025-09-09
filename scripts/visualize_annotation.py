import imageio
import json
import matplotlib.pyplot as plt

anno_path = 'lerobot/realman/stir_coffee/annotations/episode_000000.json'
with open(anno_path, 'r') as f:
    annotations = json.load(f)

video_path = 'lerobot/realman/stir_coffee/videos/chunk-000/observation.images.cam_high/episode_000000.mp4'
frames = imageio.v3.imread(video_path, plugin="pyav")


plt.ion()
for frame, anno in zip(frames, annotations):
    plt.clf()
    plt.imshow(frame)
    plt.title(anno['mov_summary_left'] + ' ' + anno['mov_summary_right'])
    plt.axis('off')
    plt.pause(0.1)