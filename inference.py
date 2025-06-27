import torch
import numpy as np
from decord import VideoReader, cpu
from transformers import DistilBertTokenizer
import pandas as pd
import os
import sys
sys.path.append(os.path.relpath('./module_utils/'))
import data_utils, loss_utils, prompt_utils
from model_utils import CLIPModel, ProjectionHead2, CFG
import torch.nn.functional as F

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    end_idx = seg_len
    start_idx = 0
    L = np.linspace(start_idx, end_idx - 1, num=clip_len + 1)
    indices = []
    for i in range(len(L) - 1):
        idx = np.random.randint(L[i], L[i + 1])
        indices.append(idx)
    indices = np.clip(indices, start_idx, end_idx).astype(np.int64)
    return indices

def predict_video_class(video_path, model_path, classifier_path, label_path, gt_path):

    label_df = pd.read_csv(label_path)
    labels = list(label_df['name'].values)

    gt_label = None
    gt_df = pd.read_csv(gt_path)
    video_name = os.path.basename(video_path)
    match = gt_df[gt_df['clip'] == video_name]
    if not match.empty:
        gt_label = match.iloc[0]['caption']

    clip_model = CLIPModel().to(CFG.device)
    clip_model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    clip_model.eval()

    classifier = ProjectionHead2(embedding_dim=CFG.projection_dim, projection_dim=len(labels)).to(CFG.device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=CFG.device))
    classifier.eval()

    vr = VideoReader(video_path, ctx=cpu(0), width=CFG.width, height=CFG.height)

    if len(vr) > CFG.frame:
        indices = sample_frame_indices(CFG.frame, frame_sample_rate=1, seg_len=len(vr))
    else:
        indices = np.sort(np.random.choice(len(vr), CFG.frame, replace=True))

    clip = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
    clip = torch.tensor(clip).permute(0, 3, 1, 2).unsqueeze(0).float().to(CFG.device)  # (1, T, C, H, W)

    with torch.no_grad():
        image_features = clip_model.image_encoder(clip)
        image_features = image_features.squeeze(-1).squeeze(-1).squeeze(-1)
        image_embeddings = clip_model.image_projection(image_features)

        logits = classifier(image_embeddings)
        probs = F.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        pred_label = labels[pred_idx]
        pred_conf = probs[0, pred_idx].item()

    return pred_label, gt_label

# Example usage:
if __name__ == "__main__":
    video_path = "Sample_data/Sample0036_8358_8373.mp4"
    model_path = "CLIPmodel.pt"               # CLIPModel checkpoint
    classifier_path = "Classifier.pt"    # Classifier head checkpoint
    label_path = "Sample_data/Clip_label.csv"
    gt_path = "Sample_data/Clip.csv"

    label, gt_label = predict_video_class(video_path, model_path, classifier_path, label_path, gt_path)
    print(f"Predicted class: {label}, Ground Truth class: {gt_label}")
