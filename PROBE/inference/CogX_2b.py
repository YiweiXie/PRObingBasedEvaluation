import os
import subprocess
import argparse

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    prompt = "nudity"
    
    # # 2B configuration
    model_path = "THUDM/CogVideoX-2b"
    eraser_rank = args.eraser_rank
    num_frames = args.num_frames
    seed = args.seed    

    eraser_path = "../weight/CogX2b_nudity"
    output_path = args.output_path
    command = [
        "python",
        args.test_script,
        f"--prompt={prompt}",
        f"--model_path={model_path}",
        f"--eraser_path={eraser_path}",
        f"--eraser_rank={eraser_rank}",
        f"--num_frames={num_frames}",
        f"--output_path={output_path}",
        f"--seed={seed}",
        f"--embed_path={args.embed_path}",
        f"--neg_prompt={args.neg_prompt}", 
        f"--csv_path={args.csv_path}"
    ]

    if args.generate_clean:
        command.append("--generate_clean")
    if args.generate_neg:
        command.append("--generate_neg")
    if args.generate_probe:
        command.append("--generate_probe")

    print(f"Running command:\n{' '.join(command)}\n")
    
    subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="nudity")
    parser.add_argument("--neg_prompt", type=str, default="nudity")
    parser.add_argument("--embed_path", type=str, default="")
    parser.add_argument("--eraser_path", type=str, default="../weight/CogX2b_nudity")
    parser.add_argument("--eraser_rank", type=int, default=128)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_script", type=str, default="test_cogvideo_negprompt_probe.py")
    parser.add_argument("--output_path", type=str, default="../output")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--generate_clean", type=bool, default=False)
    parser.add_argument("--generate_neg", type=bool, default=False)
    parser.add_argument("--generate_probe", type=bool, default=False)

    args = parser.parse_args()

    main(args=args)