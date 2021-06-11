# Identity and Attribute Preserving Thumbnail Upscaling
Accompanying Code for the Accepted ICIP 2021 Paper [IDENTITY AND ATTRIBUTE PRESERVING THUMBNAIL UPSCALING](https://arxiv.org/abs/2105.14609) by Noam Gat, Sagie Benaim and  Lior Wolf.

## Abstract
We consider the task of upscaling a low resolution thumbnail image of a person, to a higher resolution image, which preserves the person's identity and other attributes. Since the thumbnail image is of low resolution, many higher resolution versions exist. Previous approaches produce solutions where the person's identity is not preserved, or biased solutions, such as predominantly Caucasian faces. We address the existing ambiguity by first augmenting the feature extractor to better capture facial identity, facial attributes (such as smiling or not) and race, and second, use this feature extractor to generate high-resolution images which are identity preserving as well as conditioned on race and facial attributes. Our results indicate an improvement in face similarity recognition and lookalike generation as well as in the ability to generate higher resolution images which preserve an input thumbnail identity and whose race and attributes are maintained. 

## Architecture

### Face Feature Extraction
We build upon the face feature extraction architecture of [ArcFace](https://arxiv.org/abs/1801.07698), and add additional training losses to explicitly preserve identity, race and facial attributes:
![Face Feature Extraction](images/FeatureExtraction.png "Face Feature Extraction")
Note that the feature extraction network is the same one with the same weights for all tasks. 

### Face Generation
We then use this augmented feature extractor to generate faces with a technique similar to [PULSE](https://arxiv.org/abs/2003.03808), with additional user-provided goals to control the identity of the generated faces:
![Face Generation](images/FaceGeneration.png "Face Generation") 

## Results

### Similar Identity
![Similar Identity](images/SimilarIdentity.png "Similar Identity")

### Implicit and Explicit Ethnic Target
![Implicit and Explicit Ethnic Target](images/ObamaExperiment.jpg "Implicit and Explicit Ethnic Target")

### Explicit Attribute Target
![Attribute Target](images/AttributeControl.png "Attribute Target")

## Running Instructions
TODO

## Training instructions
1. Download ms1m-retinaface-t1.zip from https://github.com/deepinsight/insightface/wiki/Dataset-Zoo , place in InsightFace_v2/data 
2. run `InsightFace_v2/extract.py`
3. Install conda environment from `preprocess.yml`
4. run `python pre_process.py` from InsightFace_v2 directory

TODO