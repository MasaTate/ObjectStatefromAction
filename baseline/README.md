# CLIP Baseline Evaluation on MOST Dataset

This repository contains the evaluation code for the CLIP baseline on the **MOST Dataset**.  
You can experiment with different text prompts and measure their performance.

---

## 🚀 Get Started

### Requirements

Make sure you have the following environment:
```bash
python==3.8
torch==1.11.0
clip==1.0
```


## 📥 Download Weights
Download clip weights with
```
mkdir -p ./weights && wget -P ./weights/ https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
```

## 🧪 Run Evaluation
Run the evaluation script with a custom prompt template:
```
python eval_clip_zeroshot.py --prompt_prefix "a photo of state object" --save_dir ./results/1fps/clip --fps 1
```

The placeholders "state" and "object" will automatically be replaced with the state label and object name, respectively.
Example: "a photo of state object" → "a photo of whole apple"

You can also use detailed descriptions as prompts by providing a directory path containing state description files:
```
python eval_clip_zeroshot.py --prompt_prefix "../MOST_dataset/descriptions" --save_dir ./results/1fps/clip --fps 1
```