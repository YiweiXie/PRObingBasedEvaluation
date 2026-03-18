import subprocess
import os
import argparse

dimension = "overall_consistency"  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VBench', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--videos_path', type=str, default=None, help='path to videos')
    parser.add_argument('--output_path', type=str, default=None, help='path to output')
    args = parser.parse_args()


    cmd = [
        "python", "./src/VBench-master/evaluate.py",
        "--output_path", args.output_path,
        "--dimension", dimension,
        "--videos_path", args.videos_path,
        "--mode=custom_input",
    ]

    subprocess.run(cmd)
    print("Successfully run!")