import os
import sys
import clip
import torch
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from src.dataset import MultiClassFromFeatEval
from src.metrics import MultiClassEvaluator
from src.model import  ClipTextEncoder


def main(args):

    print("load dataset")
    video_ids = args.video_ids
    dataset = MultiClassFromFeatEval(
        dataset_root=args.data_root,
        categories=args.categories,
        fps=args.fps,
        backbone="ViT_L_14",
        video_ids=video_ids,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    evaluator = MultiClassEvaluator(
        data_root=args.annot_root,
        dict_root=args.state_list_dir,
        categories=args.categories,
        fps=args.fps,
        video_ids=video_ids,
        threshold_list=None,
    )

    device = torch.device(
        f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"
    )
    weight = torch.jit.load("weights/ViT-L-14.pt", map_location="cpu").state_dict()

    model_text = ClipTextEncoder(
        params=weight,
        train_backbone=False,
    )

    model_text.to(device)
    model_text.eval()

    for i, (category, video_id, feats) in enumerate(tqdm(dataloader)):
        category = category[0]
        video_id = video_id[0]
        feats = feats[0]

        print(category, video_id, flush=True)

        # text features for each state category
        if args.prompt_prefix.split("/")[-1] == "descriptions":
            with open(
                os.path.join(
                    args.prompt_prefix, f"{category}", f"{category}_def_v1.json"
                )
            ) as f:
                descriptions = json.load(f)
                # print(descriptions)
            prompt_list = []
            for state in evaluator.states[category]:
                if f"The {category} is {state}" in descriptions.keys():
                    prompt_list.append(
                        (" ").join(
                            descriptions[f"The {category} is {state}"].split(" ")[:55]
                        )
                    )
                else:
                    prompt_list.append(f"The {category} is {state}")
        else:
            prompt_list = [
                f"{args.prompt_prefix.replace('object', category).replace('state', state)}"
                for state in evaluator.states[category]
            ]
        # print(prompt_list)
        tokenized = clip.tokenize(prompt_list).to(device)
        with torch.no_grad():
            text_feats = model_text(tokenized).cpu()
            text_feats /= text_feats.norm(dim=-1, keepdim=True)

        vision_feats = feats / feats.norm(dim=-1, keepdim=True)

        scores = (text_feats @ vision_feats.T).numpy()

        # print(text_feats.shape)
        # print(vision_feats.shape)
        # print(scores)

        evaluator.evaluate_video_fast(category, video_id, scores)

    metrics = evaluator.calc_metrics()
    # print(metrics)
    # print(evaluator.get_num_eval_frames())

    for cat in args.categories:
        print(f"\n=== {cat} ===")
        ap = metrics["AP"][cat]
        pr_auc = metrics["PR_AUC"][cat]
        f1_max = metrics["F1_max"][cat]
        f1_10 = metrics["F1_10"][cat]
        f1_20 = metrics["F1_20"][cat]
        f1_30 = metrics["F1_30"][cat]

        print("\n===AP===")
        ap_list = []
        for k, v in ap.items():
            ap_list.append(v)
            print(f"{k},{v}")

        print("\n===PR-AUC===")
        state_list = []
        pr_auc_list = []
        for k, v in pr_auc.items():
            state_list.append(k)
            pr_auc_list.append(v)
            print(f"{k},{v}")

        print("\n===F1-MAX===")
        f1_max_list = []
        for k, v in f1_max.items():
            f1_max_list.append(v)
            print(f"{k},{v}")

        print("\n==F1-10==")
        f1_10_list = []
        for k, v in f1_10.items():
            f1_10_list.append(v)
            print(f"{k},{v}")

        print("\n==F1-20==")
        f1_20_list = []
        for k, v in f1_20.items():
            f1_20_list.append(v)
            print(f"{k},{v}")

        print("\n==F1-30==")
        f1_30_list = []
        for k, v in f1_30.items():
            f1_30_list.append(v)
            print(f"{k},{v}")

        save_df = pd.DataFrame(
            {
                "state": state_list,
                "ap": ap_list,
                "pr_auc": pr_auc_list,
                "f1_max": f1_max_list,
                "f1_10": f1_10_list,
                "f1_20": f1_20_list,
                "f1_30": f1_30_list,
            }
        )
        if args.prompt_prefix.split("/")[-1] == "descriptions":
            save_cat_dir = os.path.join(
                args.save_dir, args.prompt_prefix.split("/")[-1], cat
            )
        else:
            save_cat_dir = os.path.join(
                args.save_dir, args.prompt_prefix.replace(" ", "_"), cat
            )
        os.makedirs(save_cat_dir, exist_ok=True)
        save_path = os.path.join(save_cat_dir, f"{cat}.csv")
        save_df.to_csv(save_path, index=False)

        # plot PR curve
        precision_list = metrics["PR_list"][cat]
        recall_list = metrics["RE_list"][cat]
        for k, v in precision_list.items():
            save_path = os.path.join(save_cat_dir, f"{cat}_{k.replace(' ','_')}.png")
            plot_pr_curve(v, recall_list[k], k, save_path)


def plot_pr_curve(precision: list, recall: list, category: str, save_path: str):
    """
    Plots the PR (Precision-Recall) curve.

    Args:
    - precision: List of precision values.
    - recall: List of recall values.
    - category: Name of the category to be displayed in the plot title.
    """

    # Ensure precision and recall lists have the same length
    assert len(precision) == len(
        recall
    ), "Precision and recall lists must have the same length."

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker=".", color="b", lw=2)
    plt.fill_between(recall, precision, color="skyblue", alpha=0.4)
    plt.title(f"Precision-Recall Curve for {category}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid(True)
    plt.savefig(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--annot_root",
        type=str,
        default="../MOST_dataset/test_annotations",
    )
    parser.add_argument(
        "--state_list_dir",
        type=str,
        default="../MOST_dataset/state_category",
        )
    parser.add_argument(
        "--categories",
        nargs="+",
        type=str,
        default=["apple", "egg", "flour", "shirt", "tire", "wire"],
    )
    parser.add_argument("--cuda_num", type=int, default=0)
    parser.add_argument("--prompt_prefix", type=str, default="The object is state")
    parser.add_argument("--save_dir", type=str, default="./results/clip")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--video_ids", nargs="+", default=None)

    args = parser.parse_args()
    main(args)
