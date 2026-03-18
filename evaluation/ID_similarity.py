import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    face_tensor = mtcnn(img)
    if face_tensor is None:
        return None
    if face_tensor.ndim == 3:  # [3,H,W] → [1,3,H,W]
        face_tensor = face_tensor.unsqueeze(0)
    face_tensor = face_tensor.to(device)
    with torch.no_grad():
        emb = model(face_tensor)
    return emb.cpu().numpy()[0]  # return 1D embedding

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_dir', type=str, default='./data/reference_set')
    parser.add_argument('--test_dir', type=str, default="./generate_data/cog2bX/ID/Barack_Obama")
    parser.add_argument('--output_csv', type=str, default='./eval/ID-Similarity/similarity_results_obama.csv')
    args = parser.parse_args()
    
    ref_dir = args.ref_dir
    test_dir = args.test_dir

    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    reference_embeddings = {}
    for person in os.listdir(ref_dir):
        pdir = os.path.join(ref_dir, person)
        if not os.path.isdir(pdir):
            continue
        embs = []
        for fn in os.listdir(pdir):
            if fn.lower().endswith(('jpg','jpeg','png')):
                path = os.path.join(pdir, fn)
                emb = extract_embedding(path)
                if emb is not None:
                    embs.append(emb)
        if embs:
            reference_embeddings[person] = np.mean(embs, axis=0)
            print(f"Loaded {len(embs)} images for {person}")

    results = []
    for fn in os.listdir(test_dir):
        if not fn.lower().endswith(('jpg','jpeg','png')):
            continue
        test_path = os.path.join(test_dir, fn)
        emb = extract_embedding(test_path)
        if emb is None:
            print(f"Skipped {fn}, no face detected")
            default_name = list(reference_embeddings.keys())[0]
            results.append((fn, default_name, 0))
            print(f"{fn} → {default_name} (sim=0.0000) - No face detected")
            continue
        sims = {name: cosine_similarity(emb, ref_emb) 
                for name, ref_emb in reference_embeddings.items()}
        best_match = max(sims.items(), key=lambda x: x[1])
        results.append((fn, best_match[0], best_match[1]))
        print(f"{fn} → {best_match[0]} (sim={best_match[1]:.4f})")


    df = pd.DataFrame(results, columns=['test_image', 'best_match', 'similarity'])
    df.to_csv(args.output_csv, index=False)
    mean_similarity = df['similarity'].mean()
    print(f"\n mean_similarity: {mean_similarity:.4f}")
    print(f"Done. Results saved to {args.output_csv}")
