import os
import imageio
import argparse

if __name__ == "__main__":

    def mp4_to_png_flat(video_path, output_dir, video_id):
        os.makedirs(output_dir, exist_ok=True)
        reader = imageio.get_reader(video_path)

        for i, frame in enumerate(reader):
            frame_name = f"{video_id}_{i}.png"
            frame_path = os.path.join(output_dir, frame_name)
            imageio.imwrite(frame_path, frame)

        print(f"✅ Finish: {video_path} → {i+1} frames")

    def process_folder_flat(input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for filename in os.listdir(input_folder):
            if filename.endswith(".mp4"):
                video_path = os.path.join(input_folder, filename)
                video_id = os.path.splitext(filename)[0]
                mp4_to_png_flat(video_path, output_folder, video_id)

    parser = argparse.ArgumentParser(description="Process video files.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing video files.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder to save PNG files.")
    args = parser.parse_args()

    process_folder_flat(args.input_folders, args.output_folders)