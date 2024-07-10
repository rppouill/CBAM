# CBAM: Convolutional Block Attention Module [[1](https://arxiv.org/abs/1807.06521)] (Tensorflow) 




## Channel Attention Module (CAM)
To generate a channel attention map by leveraging the inter-channel relationship, we aim to identify 'what' is meaningful in a given input image. 
The process of computing the channel attetion map is as follows:
$$M_c(F) = \sigma(\text{MLP}(AvgPool(F)) + \text{MLP}(MaxPool(F)))$$
where $MLP$ is a Multi Perceptron Layer.
<p align="center">
  <img width="696" height="175" src="img/diagrams/cam_diagram.jpg">
</p>

## patial Attention Module (SAM)
To generate a spatial attention map using the inter-spation relationship, we aim to identify 'where' is the informative part.
The process of computing the spatial attention map is as follows:
$$M_s(F) = \sigma(f^{n*n}([AvgPool(F);MaxPool(F)]))$$
where $f^(n*n)$ is a layer convolution size of $n*n$.
<p align="center">
  <img width="579" height="209" src="img/diagrams/sam_diagram.jpg">
</p>

## Convolution Block Attention Module (CBAM)
The Convolutional Block Attention Module (CBAM) combines the Channel Attention Module (CAM) and the Spatial Attention Module (SAM) to produce a comprehensive attention map that includes both channel and spatial attention.

CBAM is computing as follows:
$$ F' = M_c(F) \otimes F 
    F'' = M_s(F') \otimes F' $$
where $\otimes$ denotes element-wise multiplication.
<p align="center">
  <img width="683" height="212" src="img/diagrams/cbam_diagram.jpg">
</p>


## References
[[1](https://arxiv.org/abs/1807.06521)] Sanghyun Woo,Jongchan Park, Joon-Young Lee, and In So Kweon. CBAM: Convolutional Block Attention Module. 2018.
