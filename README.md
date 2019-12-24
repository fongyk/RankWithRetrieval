# Rank With Retrieval
image retrieval with different losses and datasets

# requirements

pytorch: 0.3.0

python: 2.7.x

# datasets

## testing

- Ukbench

  Scalable recognition with a vocabulary tree, cvpr 2006

- Oxford5k

  https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/
  
- Parid6k

  https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/

- LFW face

  http://vis-www.cs.umass.edu/lfw/

## training

- landmark_clean

  for Oxford5k & Parid6k

  https://europe.naverlabs.com/Research/Computer-Vision/Learning-Visual-Representations/Deep-Image-Retrieval/

- CASIA-Webface

  for LFW face

# losses

- RankNet loss 

  (2005-Learning to rank using gradient descent-icml)

- triplet loss

- contrastive loss

- softmax loss

- n-pair loss 

  2016-Improved Deep Metric Learning with Multi-class N-pair Loss Objective-nips


