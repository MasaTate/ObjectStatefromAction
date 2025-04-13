import os
import glob
import torch
import numpy as np
import torch.utils.data as data


class MultiClassFromFeatEval(data.Dataset):
    def __init__(
        self, dataset_root, categories, fps=10, backbone="ViT_L_14", video_ids=None
    ):
        self.root = dataset_root
        self.category = categories
        self.fps = fps
        if video_ids:
            self.video_ids = video_ids

        self.video_feats_list = []
        for cat in categories:
            for p in glob.glob(
                os.path.join(dataset_root, "frame_feat", cat, backbone, "*")
            ):
                if video_ids:
                    print(video_ids)
                    if os.path.basename(p).split(".")[0] not in video_ids:
                        continue
                
                feat = np.load(p).copy()
                if self.fps != 10:
                    indices = np.arange(0, feat.shape[0], 10//self.fps)
                    feat = feat[indices]
                self.video_feats_list.append(
                    {
                        "category": cat,
                        "video_id": os.path.basename(p).split(".")[0],
                        "path": p,
                        "feat": feat,
                    }
                )

    def __getitem__(self, index):
        category = self.video_feats_list[index]["category"]
        video_id = self.video_feats_list[index]["video_id"]
        feat = torch.from_numpy(self.video_feats_list[index]["feat"].copy()).float()

        return category, video_id, feat

    def __len__(self):
        return len(self.video_feats_list)