# Training data sampling options

## Sampling for multiple labels
In any case when corresponding labels are available and there are multiple types of labels, e.g. the segmentation of different organs in a CT image, two options are available:

- During one epoch, each image would be sampled only once and when there are multiple labels, we will randomly sample one label at a time. (Default) 
- During one epoch, each image would be paired with each available label. So if an image has four types of labels, it will be sampled for four times and each time corresponds to a different label. When using multiple labels, it is the user's responsibility to ensure the labels are ordered, such that label_idx are the corresponding types in (width, height, depth, label_idx)

## Sampling for multiple subjects each with multiple images
When multiple subjects each with multiple images are available, multiple different sampling methods are supported:

- Inter-subject, one image is sampled from subject A as moving image, and another one image is sampled from a different subject B as fixed image.   
- Intra-subject, two images are sampled from the same subject. In this case, we can specify:  
    - a) moving image always has a smaller index, e.g. at an earlier time;  
    - b) moving image always has a larger index, e.g. at a later time; or  
    - c) no constraint on the order.  
For the first two options, the intra-subject images will be ascending-sorted by name to represent ordered sequential images, such as time-series data *Multiple label sampling is also supported once image pair is sampled;  
In case there are no consistent label types defined between subjects, an option is available to turned off label contribution to the loss for those inter-subject image pairs.
