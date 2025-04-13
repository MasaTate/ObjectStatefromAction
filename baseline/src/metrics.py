import os
import glob
import numpy as np
import pandas as pd
from scipy.integrate import trapz as integral
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.metrics import f1_score


class MultiClassEvaluator:
    def __init__(
        self,
        data_root: str,
        categories: list,
        fps: int,
        video_ids=None,
        dict_root=None,
        direct_dict=None,
        threshold_list=None,
        binary_prediction=False,
    ):
        self.data_root = data_root
        self.dict_root = dict_root
        self.categories = categories
        self.binary_prediction = binary_prediction
        self.states = {}
        self.fps = fps
        self.annotation_paths = {}
        self.result = {}
        """ 
        low_thresholds = np.arange(0, 20, 2) / 100
        middle_thresholds = np.arange(20, 80, 5) / 100
        high_thresholds = np.arange(8, 11, 1) / 10
        self.thresholds = np.concatenate(
            [low_thresholds, middle_thresholds, high_thresholds]
        )
        """
        self.thresholds = np.arange(0, 1001, 1) / 1000
        if threshold_list is not None:
            self.thresholds = threshold_list
        print("thresholds: ", self.thresholds)
        self.metrics = ["precision", "recall"]

        assert dict_root is not None or direct_dict is not None
        self.video_ids = []
        for cat in categories:
            # Load state dictionary
            if dict_root is not None:
                state_dict_path = os.path.join(dict_root, cat + ".txt")
                with open(state_dict_path, "r") as f:
                    states = f.readlines()
            if direct_dict is not None:
                states = direct_dict[cat]
            states = [state.strip() for state in states]
            self.states[cat] = states

            # Load annotation paths
            cat_paths = glob.glob(os.path.join(data_root, cat, "*.csv"))
            self.video_ids += [
                os.path.basename(path).split(".")[0] for path in cat_paths
            ]
            self.annotation_paths[cat] = cat_paths

            # Initialize result dictionary
            self.result[cat] = {}

        if video_ids is not None:
            self.video_ids = video_ids

    def get_num_eval_frames(self):
        num_frames = {}
        for cat in self.categories:
            num_frames[cat] = {}
            for video_id in self.video_ids:
                for path in self.annotation_paths[cat]:
                    if video_id in path:
                        annotation = pd.read_csv(
                            path, dtype={"state": str, "start": float, "end": float}
                        )
                        for state in self.states[cat]:
                            state_annotation = annotation[annotation["state"] == state]
                            for _, row in state_annotation.iterrows():
                                if row["state"] == state:
                                    num_frames[cat][state] = num_frames[cat].get(
                                        state, 0
                                    ) + int((row["end"] - row["start"]) * self.fps)

        return num_frames

    def evaluate_video_fast(self, category: str, video_id: str, prediction: np.ndarray):
        assert category in self.categories
        assert prediction.shape[0] == len(self.states[category])

        # Load annotation
        annotation = next(
            (
                pd.read_csv(path, dtype={"state": str, "start": float, "end": float})
                for path in self.annotation_paths[category]
                if video_id in path
            ),
            None,
        )

        if annotation is None:
            raise ValueError(
                f'Annotation of video {video_id} in category "{category}" not found'
            )

        target = np.zeros_like(prediction)
        for i, state in enumerate(self.states[category]):
            state_annotation = annotation[annotation["state"] == state]
            for _, row in state_annotation.iterrows():
                target[i, int(row["start"] * self.fps) : int(row["end"] * self.fps)] = 1

        # Initialize result of video
        self.result[category].setdefault("predictions", [])
        self.result[category]["predictions"].append(prediction)
        self.result[category].setdefault("targets", [])
        self.result[category]["targets"].append(target)

    def get_target(self, category: str, video_id: str, length: int):
        assert category in self.categories

        # Load annotation
        annotation = next(
            (
                pd.read_csv(path, dtype={"state": str, "start": float, "end": float})
                for path in self.annotation_paths[category]
                if video_id in path
            ),
            None,
        )

        if annotation is None:
            raise ValueError(
                f'Annotation of video {video_id} in category "{category}" not found'
            )

        target = np.zeros((len(self.states[category]), length))
        for i, state in enumerate(self.states[category]):
            state_annotation = annotation[annotation["state"] == state]
            for _, row in state_annotation.iterrows():
                target[i, int(row["start"] * self.fps) : int(row["end"] * self.fps)] = 1

        return target

    def calc_metrics(self):
        # Calculate AP for each state in each category
        AP_for_each_category = {}
        PR_AUC_for_each_category = {}
        Precision_list_for_each_category = {}
        Recall_list_for_each_category = {}
        F1_max_list_for_each_category = {}
        F1_10_list_for_each_category = {}
        F1_20_list_for_each_category = {}
        F1_30_list_for_each_category = {}
        for cat in self.categories:
            targets = np.concatenate(self.result[cat]["targets"], axis=1)
            predictions = np.concatenate(self.result[cat]["predictions"], axis=1)

            # Calculate AP for each state
            AP_for_each_state = {}
            PR_AUC_for_each_state = {}
            Precision_list_for_each_state = {}
            Recall_list_for_each_state = {}
            F1_max_list_for_each_state = {}
            F1_10_list_for_each_state = {}
            F1_20_list_for_each_state = {}
            F1_30_list_for_each_state = {}

            for i, state in enumerate(self.states[cat]):
                # Calculate AP for each state
                precision, recall, _ = precision_recall_curve(
                    targets[i, :], predictions[i, :]
                )
                AP_for_each_state[state] = average_precision_score(
                    targets[i, :], predictions[i, :]
                )
                PR_AUC_for_each_state[state] = auc(recall, precision)
                Precision_list_for_each_state[state] = precision
                Recall_list_for_each_state[state] = recall
                if self.binary_prediction:
                    # threshold = 0.5
                    binary_pred = (predictions[i, :] >= 0.5).astype(int)
                    F1_max_list_for_each_state[state] = f1_score(
                        targets[i, :], binary_pred
                    )
                    F1_10_list_for_each_state[state] = f1_score(
                        targets[i, :], binary_pred
                    )
                    F1_20_list_for_each_state[state] = f1_score(
                        targets[i, :], binary_pred
                    )
                    F1_30_list_for_each_state[state] = f1_score(
                        targets[i, :], binary_pred
                    )
                else:
                    F1_max_list_for_each_state[state] = np.max(
                        2 * precision * recall / (precision + recall + 1e-10)
                    )

                    binary_pred_10 = (predictions[i, :] >= 0.1).astype(int)
                    F1_10_list_for_each_state[state] = f1_score(
                        targets[i, :], binary_pred_10
                    )

                    binary_pred_20 = (predictions[i, :] >= 0.2).astype(int)
                    F1_20_list_for_each_state[state] = f1_score(
                        targets[i, :], binary_pred_20
                    )

                    binary_pred_30 = (predictions[i, :] >= 0.3).astype(int)
                    F1_30_list_for_each_state[state] = f1_score(
                        targets[i, :], binary_pred_30
                    )

            AP_for_each_category[cat] = AP_for_each_state
            PR_AUC_for_each_category[cat] = PR_AUC_for_each_state
            Precision_list_for_each_category[cat] = Precision_list_for_each_state
            Recall_list_for_each_category[cat] = Recall_list_for_each_state
            F1_max_list_for_each_category[cat] = F1_max_list_for_each_state
            F1_10_list_for_each_category[cat] = F1_10_list_for_each_state
            F1_20_list_for_each_category[cat] = F1_20_list_for_each_state
            F1_30_list_for_each_category[cat] = F1_30_list_for_each_state

        return {
            "AP": AP_for_each_category,
            "PR_AUC": PR_AUC_for_each_category,
            "PR_list": Precision_list_for_each_category,
            "RE_list": Recall_list_for_each_category,
            "F1_max": F1_max_list_for_each_category,
            "F1_10": F1_10_list_for_each_category,
            "F1_20": F1_20_list_for_each_category,
            "F1_30": F1_30_list_for_each_category,
        }