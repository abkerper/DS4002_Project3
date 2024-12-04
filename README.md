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
    - caltech_images/
        - attributes/
            - certainties.txt
            - class_attribute_labels_continuous.txt
            - image_attribute_labels.txt
        - images/
            - 001.Black_footed_Albatross
                - Roughly 60 jpg images of Black footed ALbatross birds
            - 002.Laysan_Albatross
                - Roughly 60 jpg images of Laysan Albatross birds
            - 003.Sooty_Albatross
                - Roughly 60 jpg images of Scooty Albatross birds
            - 004.Groove_billed_Ani
                - Roughly 60 jpg images of Groove biiled Ani birds
            - 005.Crested_Auklet
                - Roughly 60 jpg images of Crested Auklet birds
            - 006.Least_Auklet
                - Roughly 60 jpg images of Least AUklet birds
            - 007.Parakeet_Auklet
                - Roughly 60 jpg images of Parakeet Auklet birds
            - 008.Rhinoceros_Auklet
                - Roughly 60 jpg images of Rhinocerous Auklet birds
            - 009.Brewer_Blackbird
                - Roughly 60 jpg images of Brewer Blackbirds
            - 012.Yellow_headed_Blackbird
                - Roughly 60 jpg images of Yellow headed blackbirds
            - 013.Bobolink
                - Roughly 60 jpg images of Bobolink birds
            - 014.Indigo_Bunting
                - Roughly 60 jpg images of Indigo Bunting birds
            - 019.Gray_Catbird
                - Roughly 60 jpg images of Gray Catbirds
        - parts/
            - part_click_locs.txt
            - part_locs.txt
            - parts.txt
            - attributes.txt
            - bounding_boxes.txt
            - classes.txt
            - image_class_labels.txt
            - images.txt
            - train_test_split.txt
        - avonet_data.csv
        - DataProcess.md
- SCRIPTS/
    - analysis.ipynb
    - datasets.py
    - ratio.ipynb
    - sam_segmentation.py
- OUTPUT/
    - bird_beak_to_body_medians.csv
    - bird_beak_to_body_ratios.csv
    - estimated_beak_lengths_comparison.csv
    - RESULTS.pdf
- README.md
- LISCENSE.md

## Instructions for Reproducing
- Follow the installation instructions from [SAM](https://github.com/facebookresearch/segment-anything)
- Download the SAM model, [vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
    - NOTE THIS WILL AUTOMATICALLY DOWNLOAD WHEN CLICKED (2.4 GB TOTAL)
- Generate segmentation masks using SAM by running 
```{python}
python3 SCRIPTS/sam_segmentation.py --prompt=point --model=vit_h
```
    - Note: This may take a while (poetntially several hours to go through all images)
- Then with the collected data, run the ratio.ipynb notebook to create the bird_beak_to_body_ratios.csv and bird_beak_to_body_median.csv in the OUTPUT/ directory
    - bird_beak_to_body_ratios.csv contains the average beak to body ratio for each bird analyzed
    - bird_beak_to_body_median.csv contains the median beak to body ratio for each bird analyzed
- Run analysis.ipynb to obtain the results found in the OUTPUT/ directory of this repository 

## References
[1]"Macaulay Library," Macaulay Library, [Online]. Available: https://www.macaulaylibrary.org/. [Accessed: Nov. 10, 2024].

[2] C. Wah, S. Branson, P. Welinder, P. Peronaand S. Belongie, “The Caltech-UCSD Birds-200-2011 Dataset”, California Institute of Technology, Jul. 2011.

[3] mmmmimic, “GitHub - mmmmimic/CUB-SAM: Segmenting Birds with SAM,” GitHub, 2023. https://github.com/mmmmimic/CUB-SAM (accessed Dec. 04, 2024).

[2]A. Kirillov et al., “Segment Anything,” GitHub, Apr. 17, 2023. https://github.com/facebookresearch/segment-anything