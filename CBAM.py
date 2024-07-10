if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import logging as log
import coloredlogs


class ChannelAttentionModule(tf.keras.layers.Layer):
    def __init__(self, channel , r, **kwargs):
        """
        Channel Attention Module
        Description:
            This module is used to capture channel-wise dependencies. 
            It uses global average pooling and global max pooling to 
            capture the channel-wise dependencies. It then uses a 
            multi-layer perceptron to generate the attention map.

        Latex Formula:
            $$ M_c(F) = \sigma(\text{MLP}(AvgPool(F)) + \text{MLP}(MaxPool(F)))$$

        Args:
            channel : number of channels
            r       : reduction ratio
        
        Returns:
            output: Channel attention map
        """
        super(ChannelAttentionModule, self).__init__(**kwargs)
        self.channel = channel
        self.r = r
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.channel//r, activation = 'relu'),
            tf.keras.layers.Dense(self.channel, activation = None)])
    
    def call(self,x):
        max = tf.reduce_max (x, axis = (1,2), keepdims=True)
        avg = tf.reduce_mean(x, axis = (1,2), keepdims=True)
        linear_max = self.mlp(max)
        linear_avg = self.mlp(avg)
        output = linear_max + linear_avg
        output = tf.tile(tf.expand_dims(tf.expand_dims(tf.keras.activations.sigmoid(output), axis=1), axis=2), [1, tf.shape(x)[1], tf.shape(x)[2], 1])

        return output * x
    

class SpatialAttentionModule(tf.keras.layers.Layer):
    def __init__(self, bias = False, **kwargs):
        """
        Spatial Attention Module
        Description:
            This module is used to capture spatial dependencies. 
            It uses global average pooling and global max pooling to 
            capture the spatial-wise dependencies. It then uses a 
            convolutional layer to generate the attention map.

        Formula:
            $$ M_s(F) = \sigma(Conv(Concat(MaxPool(F), AvgPool(F))))$$
        
        Args:
            bias: Boolean value to use bias in convolutional layer
        returns:
            output: Spatial attention map
        """
        super(SpatialAttentionModule, self).__init__(**kwargs)
        self.bias = bias
        self.conv = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=1, padding='same', 
                                           use_bias=self.bias, activation='sigmoid')#, kernel_initializer = 'zeros')
    
    def call(self,x):
        max = tf.reduce_max (x, axis = -1, keepdims=True)
        avg = tf.reduce_mean(x, axis = -1, keepdims=True)
        concat = tf.concat([max, avg], axis = -1)
        output = self.conv(concat)
        output *= x

        return output
    

class CBAM(tf.keras.layers.Layer):
    def __init__(self, channels, r, bias = False, **kwargs):
        """
        Convolutional Block Attention Module
        
        Description:
            This module is used to capture channel-wise and spatial-wise dependencies. 
            It uses Channel Attention Module and Spatial Attention Module to generate 
            the attention map.
        
        Formula:
            $$ F' = M_c(F) \otimes F 
               F'' = M_s(F') \otimes F' $$
        
        Args:
            channels: number of channels
            r       : reduction ratio
            bias    : Boolean value to use bias in convolutional layer
        
        Returns:
            output: Attention map
        
        """
        super (CBAM, self).__init__(**kwargs)
        self.channels = channels
        self.r = r
        self.SAM = SpatialAttentionModule(bias)
        self.CAM = ChannelAttentionModule(self.channels, self.r)
    
    def call(self,x):
        output = self.CAM(x)
        output = self.SAM(output)
        return output

def main(filename, channel, r):
    def test_CAM(img, channels, r):
        cam = ChannelAttentionModule(channels, r)
        y = cam(img)
        log.info(f"Shape of CAM output: {y.shape} - Type: {type(y)}")


        return y

    def test_SAM(img):
        sam = SpatialAttentionModule()
        y = sam(img)
        log.info(f"Shape of SAM output: {y.shape} - Type: {type(y)}")

        return y

    def test_CBAM(img, channels, r):
        cbam = CBAM(channels, r)
        y = cbam(img)
        log.info(f"Shape of CBAM output: {y.shape} - Type: {type(y)}")

        return y
    
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    resize = (224,224)
    while img.shape[0] > 224:
        img = cv2.pyrDown(img)
    img = cv2.resize(img, resize)
    img = np.reshape(img.astype(np.float32) / 255.0, (1, resize[0], resize[1], 1))
    log.info(f"Shape of input image: {img.shape} - Type: {type(img)}")

    img_CAM  = np.array((test_CAM (img,16,4)  * 255)).astype(np.uint8)
    img_SAM  = np.array((test_SAM (img)       * 255)).astype(np.uint8)
    img_CBAM = np.array((test_CBAM(img,16,4)  * 255)).astype(np.uint8)
if __name__ == '__main__':
    import cv2
    import argparse

    parser = argparse.ArgumentParser(description='CBAM Script')
    parser.add_argument('--verbose', type=int, default=20, help='Logging level')
    parser.add_argument('--filename', type=str, default='img/cat.jpg', help='Image filename')
    parser.add_argument('--channel', type=int, default=16, help='Number of channels')
    parser.add_argument('--r', type=int, default=4, help='Reduction ratio')

    args = parser.parse_args()

    log.basicConfig(level=args.verbose)
    logger = log.getLogger(__name__)
    fmt = '%(asctime)s [%(process)d] %(filename)s:%(lineno)d %(levelname)s - %(message)s'
    field = {'asctime'  : {'color': 'green'}, 
             'process'  : {'color': 'magenta'}, 
             'filename' : {'color': 'yellow'}, 
             'lineno'   : {'color': 'yellow'}, 
             'levelname': {'color': 'cyan', 'bold': True}, 
             'message'  : {'color': 'black'}}
    coloredlogs.install(level=args.verbose, logger = logger, fmt = fmt, 
                        field_styles = field)

    main(args.filename, args.channel, args.r)







