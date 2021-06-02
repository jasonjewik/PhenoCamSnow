# Snow Classifier

**Goal:** To classify images taken at the [BERMS Old Jack Pine Site, Saskatchewan, Canada](https://phenocam.sr.unh.edu/webcam/sites/canadaojp/) (canadaojp).

## Training Data

Training data consists of RGB images taken at the canadaojp site between 08:00-20:00 local time every day of July 2016, October 2016, and January 2017 (except for 1/11-1/14).

(snow-classifier) PS C:\Users\jewik\GitRepos\snow-classifier> python classifier.py
Overall accuracy: 94.89%
No snow accuracy: 96.96%
Snow on ground accuracy: 94.98%
Snow on canopy accuracy: 86.36%

Parameters:

- nu = 0.15
- class_weight = balanced
- test_size = 0.33
- 4 color clusters

Increasing the number of input clusters to 8 doesn't signficantly improve accuracy.

(snow-classifier) PS C:\Users\jewik\GitRepos\snow-classifier> python classifier.py
Overall accuracy: 94.89%
3 Categories-->
No snow accuracy: 99.58%
Snow on ground accuracy: 95.40%
Snow on canopy accuracy: 81.72%
2 Categories-->
No snow accuracy: 99.58%
Snow anywhere accuracy: 97.89%

Parameters:

- nu = 0.15
- class_weight = balanced
- test_size = 0.33
- 4 color clusters and their percentage of the image

Interestingly, the model performs better without the ratio information.
(snow-classifier) PS C:\Users\jewik\GitRepos\snow-classifier> python classifier.py
Overall accuracy: 96.13%
3 Categories-->
No snow accuracy: 100.00%
Snow on ground accuracy: 94.17%
Snow on canopy accuracy: 91.01%
2 Categories-->
No snow accuracy: 100.00%
Snow anywhere accuracy: 98.48%

What if I try weighting each cluster by its ratio?
nope, didn't work.
