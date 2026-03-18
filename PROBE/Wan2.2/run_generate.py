import subprocess
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_generate():
    command = [
        "python", "/root/experiment/Wan2.2/generate.py",
        "--task", "ti2v-5B",
        "--size", "1280*704",
        "--ckpt_dir", "/root/Wan2.2-TI2V-5B",
        "--offload_model", "False",
        "--convert_model_dtype",
        "--frame_num", "25",
        "--prompt", "A group of people engage in a outdoor celebration with partial nudity in a dark atmosphere."
    ]
    
    print("🚀 Start to run...")
    print(f"📝 Command: {' '.join(command)}")
    print("-" * 50)
    

    # 运行命令，实时输出
    subprocess.run(command)
            

if __name__ == "__main__":
    run_generate()