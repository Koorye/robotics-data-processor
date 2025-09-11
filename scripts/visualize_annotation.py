import argparse
import imageio
import json
import matplotlib.pyplot as plt
import os


def get_default_lerobot_root():
    return os.path.expanduser('~/.cache/huggingface/lerobot')


def find_video_keys(repo_id):
    return os.listdir(os.path.join(get_default_lerobot_root(), repo_id, 'videos', 'chunk-000'))


def find_episodes(repo_id):
    with open(os.path.join(get_default_lerobot_root(), repo_id, 'meta', 'episodes.jsonl'), 'r') as f:
        lines = f.readlines()
    episodes = [json.loads(line) for line in lines]
    return episodes


def load_videos(repo_id, episode_index):
    video_keys = find_video_keys(repo_id)
    videos = {}
    chunk = 0

    for video_key in video_keys:
        chunk_path = os.path.join(get_default_lerobot_root(), repo_id, 'videos', f'chunk-{chunk:03d}')
        video_path = os.path.join(chunk_path, video_key, f'episode_{episode_index:06d}.mp4')

        while not os.path.exists(video_path):
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f'No video found for episode {episode_index} and video key {video_key}')
            chunk += 1
            chunk_path = os.path.join(get_default_lerobot_root(), repo_id, 'videos', f'chunk-{chunk:03d}')
            video_path = os.path.join(chunk_path, 'video_key', f'episode_{episode_index:06d}.mp4')
        
        frames = imageio.mimread(video_path, memtest=False)
        videos[video_key] = frames
    return videos


def load_annotation(repo_id, episode_index):
    annotation_path = os.path.join(get_default_lerobot_root(), repo_id, 'annotations', f'episode_{episode_index:06d}.json')
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    return annotation


class VideoAnnotationViewer:
    def __init__(self, videos, annotations):
        """
        初始化视频和标注查看器
        
        参数:
        videos: dict, 键为视频名称，值为帧列表
        annotations: list, 每个元素为包含标注信息的字典
        """
        self.videos = videos
        self.annotations = annotations
        self.video_names = list(videos.keys())
        self.current_frame_idx = 0
        self.num_videos = len(videos)
        
        # 计算最大帧数
        self.max_frames = max(len(frames) for frames in videos.values())
        
        # 创建图形和子图
        self.fig, self.axes = plt.subplots(1, self.num_videos, figsize=(5*self.num_videos, 5))
        if self.num_videos == 1:
            self.axes = [self.axes]
        
        # 设置键盘事件监听
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 初始化显示
        self.update_display()
        
    def on_key_press(self, event):
        """处理键盘按键事件"""
        if event.key == 'right':
            self.current_frame_idx = min(self.current_frame_idx + 1, self.max_frames - 1)
            self.update_display()
        elif event.key == 'left':
            self.current_frame_idx = max(self.current_frame_idx - 1, 0)
            self.update_display()
            
    def update_display(self):
        """更新所有子图的显示"""
        for i, name in enumerate(self.video_names):
            ax = self.axes[i]
            ax.clear()
            
            # 显示当前帧（如果存在）
            frames = self.videos[name]
            if self.current_frame_idx < len(frames):
                ax.imshow(frames[self.current_frame_idx])
            
            # 显示标注信息（如果存在且对应当前帧）
            # if self.current_frame_idx < len(self.annotations):
            #     annot = self.annotations[self.current_frame_idx]
            #     self.display_annotation_text(ax, annot)
            
            # 设置标题和帧信息
            # ax.set_title(f'{name}\nFrame {self.current_frame_idx + 1}/{self.max_frames}')
            ax.axis('off')
        
        text = self.display_annotation_text(ax, self.annotations[self.current_frame_idx])
        plt.suptitle(text, fontsize=10)

        
        plt.tight_layout()
        self.fig.canvas.draw()
    
    def display_annotation_text(self, ax, annot):
        """
        在指定轴上以文字形式显示标注信息
        
        参数:
        ax: matplotlib轴对象
        annot: 标注字典，包含各种标注信息
        """
        # 创建标注文本
        text_lines = [f"Frame {self.current_frame_idx + 1}/{self.max_frames}"]
        
        # 添加所有可用的标注字段
        for key, value in annot.items():
            if isinstance(value, list):
                # 格式化列表为字符串
                formatted_value = ", ".join([f"{v:.2f}" if isinstance(v, (int, float)) else str(v) for v in value])
                text_lines.append(f"{key}: [{formatted_value}]")
            else:
                # 格式化单个值
                formatted_value = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                text_lines.append(f"{key}: {formatted_value}")
        
        # 将文本合并为一个字符串
        text_content = "\n".join(text_lines)
        return text_content
        
        # 在图像上方添加文本标注[1,2](@ref)
        # ax.text(0.5, 1.05, text_content, transform=ax.transAxes, 
        #         fontsize=12, color='black',
        #         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'),
        #         ha='center', va='bottom', linespacing=1.5)
    
    def show(self):
        """显示可视化窗口"""
        plt.show()


def show_videos_with_annotation(videos, annotations):
    """
    显示带有文字标注的视频帧
    
    参数:
    videos: dict, 键为视频名称，值为帧列表（每个帧为NumPy数组）
    annotations: list, 每个元素为包含标注信息的字典
    """
    viewer = VideoAnnotationViewer(videos, annotations)
    viewer.show()


def main(args):
    episodes = find_episodes(args.repo_id)  # To check if the repo exists
    for episode in episodes:
        videos = load_videos(args.repo_id, episode['episode_index'])
        annotation = load_annotation(args.repo_id, episode['episode_index'])
        show_videos_with_annotation(videos, annotation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', type=str, required=True)
    args = parser.parse_args()
    main(args)
