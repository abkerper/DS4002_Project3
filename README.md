# Bird species classification based on body and beak size feature
### Allison Kerper, Elijah Kim, and Henry Chen
Hypothesis: We expect to build a model that accurately calculates the relative size of a bird’s beak from a picture of the bird.

Our data will be taken from Cornell's Macaulay Library[1]. It will contain 140 images of Downy Woodpeckers stored in a numpy array. A model will then be built to identify the bird's beak and body in each image. It will measure the length of these two parts and calculate the relative size of each beak. We will also try to replicate this process for the bird's beak and head size.

## Software and Platform
- Types of software used:
    - Python Jupyter Notebook
        - Done in Google Colab
    - Python Packages Used: argparse, DataLoader, tqdm, segment_anything, numpy, os, PIL, imgviz, torch, typing, cv2, matplotlib, csv, scipy
- Platform:

## Documentation Map
- DATA/
    - raw_images/
      - 140 .jpeg files
    - dataset.npy
- SCRIPTS/
    - image_dataset.ipynb
- OUTPUT/
- README.md
- LISCENSE.md

## Instructions for Reproducing
- Download data from Cornell's Macaulay Library [1]
- Use a for loop to iterate through each image, fixing them to a 240x240 pixel image and loading them into a numpy array
- Export dataframe and load array into github
- 

## References
[1]"Macaulay Library," Macaulay Library, [Online]. Available: https://www.macaulaylibrary.org/. [Accessed: Nov. 10, 2024].
