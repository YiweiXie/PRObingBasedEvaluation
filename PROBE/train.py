import os
import subprocess
from pathlib import Path
import argparse
os.chdir("PRObingBasedEvaluation/PROBE")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", type=str, help='which concept to train.', default="cassette player")
    parser.add_argument("--num", type=int, help='number of the concept.', default=5)
    parser.add_argument("--initializer_token", type=str, help='initializer token for the concept.', required=True)
    parser.add_argument("--erasure_model", type=str, help='erasure model.', default="None", choices=["cogvideox2b", "cogvideox5bX", "Wan2.2"])
    parser.add_argument("--learnable_property", type=str, help='learnable property.', choices=["object", "nudity"])
    parser.add_argument("--erasure_methods", type=str, help='erasure methods.', default="None", choices=["NegPrompt", "SAFREE", "T2VUnlearning"])
    parser.add_argument("--esd_checkpoint", type=str, help='esd checkpoint.', default="./T2VUnlearning/weights/cogvideox2b_nudity_erasure")
    parser.add_argument("--train_data_dir", type=str, help='train data dir.', default="./train_data/cogvideo2bX/objects/cassette_player")
    parser.add_argument("--output_dir", type=str, help='output dir.', default="./probe/cog2bX-objects_neg/")
    parser.add_argument("--neg_prompt", type=str, help='negative prompt for the concept.', required=True)
    parser.add_argument("--train_steps", type=int, help='train steps.', default=3000)
    parser.add_argument("--token_count", type=int, help='token count for the concept.', default=5)
    parser.add_argument("--i2p", action="store_true", help='whether to use i2p.') # default=False
    parser.add_argument("--i2p_metadata_path", type=str, help='i2p metadata path.', default="./i2p_metadata.json")
    args = parser.parse_args()


    run_dir = "./" + args.erasure_methods + "/"
    if args.erasure_model == "cogvideox2b":
        MODEL_NAME = "THUDM/CogVideoX-2b"
        run_dir += "CogX_2B.py"
    elif args.erasure_model == "cogvideox5bX":
        MODEL_NAME = "THUDM/CogVideoX-5b"
        run_dir += "CogX_5B.py"
    elif args.erasure_model == "Wan2.2":
        MODEL_NAME = "Wan-AI/Wan2.2-TI2V-5B"
        run_dir += "Wan2.2_5B.py"
    else:  
        raise ValueError("Invalid erasure model.")

    target_concept = args.concept 
    initializer_token = args.initializer_token
    placeholder_token = "<" + args.concept + "-object>"
        
    cmd = [
        "python",
        f"{run_dir}",
        "--pretrained_model_name_or_path", MODEL_NAME,
        "--train_data_dir", args.train_data_dir,
        "--learnable_property", args.learnable_property,
        "--placeholder_token", placeholder_token,
        "--initializer_token", args.initializer_token,
        "--neg_prompt", args.neg_prompt,
        "--resolution", "512",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--max_train_steps", str(args.train_steps),
        "--learning_rate", "2e-2",
        "--scale_lr",
        "--lr_scheduler", "cosine",
        "--lr_warmup_steps", "0",
        "--save_as_full_pipeline",
        "--checkpointing_steps", str(args.train_steps),
        "--output_dir", args.output_dir,
        "--num_train_images", "100",
        "--esd_checkpoint", args.esd_checkpoint,
        "--mixed_precision", "bf16",
        "--enable_xformers_memory_efficient_attention",
        "--num_vectors", str(args.token_count)
    ]

    if args.i2p:
        cmd.append("--i2p")
        cmd.append("--i2p_metadata_path")
        cmd.append(args.i2p_metadata_path)

    print("Command:", " ".join(cmd))    
    subprocess.run(cmd)