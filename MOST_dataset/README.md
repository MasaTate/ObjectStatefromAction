# **Multiple Object States Transition (MOST) Dataset**

## **Data Overview**

We provide two sets of annotations:  
- **Test Set**
- **Pseudo-Label Validation Set**

### **Test Set**  
The **test set** is intended for evaluating the final results of the trained models.

### **Pseudo-Label Validation Set**  
The **pseudo-label validation set** is used to validate the generated pseudo-labels during model training.


## **Dataset Contents**

The dataset is organized as follows:

- **`test_annotations/`**: Contains CSV files with annotations for the test set.
- **`pl_valid_annotations/`**: Contains CSV files with annotations for the pseudo-label validation set.
- **`state_category/`**: Contains text files listing the state categories for each object.
- **`test_video_ids.csv`**: A CSV file containing YouTube video IDs for the test set.
- **`pl_valid_video_ids.csv`**: A CSV file containing YouTube video IDs for the pseudo-label validation set.

Each annotation file includes object state categories and the corresponding positive time intervals in CSV format.


## **Video Downloads**

The videos for the dataset can be downloaded using a video downloader tool like [yt-dlp](https://github.com/yt-dlp/yt-dlp).