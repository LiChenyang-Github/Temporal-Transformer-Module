
## Temporal-Transformer-Module
The tensorflow implementation of Temporal Transformer Module (TTM), which is proposed in [Skeleton-based Gesture Recognition Using Several Fully Connected Layers with Path Signature Features and Temporal Transformer Module](https://arxiv.org/abs/1811.07081). TTM is a differentiable module to do temporal transformation (scale and translation) on the input. The transformation matrix is `[scale translation]`.


## Example 

```python
import ttm
import tensorflow as tf


batchsize = 56
final_frame_nb = 39
dim = 702   # Refer to the comment in ttm.py


X = tf.Variable(tf.random_normal([batchsize, dim]), name="input")   # X is the raw coordinates, each row is a sample in order of [joint, axis(xyz), frame].

theta = localization_net(X) # localization_net() is a network of any form (fully connected layer, 1D convolutional layer, etc.), but should finally regress to 2 neurons.  

X = temporal_transformer_network_2paras(X, final_frame_nb, theta)
```

## Citation     

    @article{li2018skeleton,
      title={Skeleton-based Gesture Recognition Using Several Fully Connected Layers with Path Signature Features and Temporal Transformer Module},
      author={Li, Chenyang and Zhang, Xin and Liao, Lufan and Jin, Lianwen and Yang, Weixin},
      journal={arXiv preprint arXiv:1811.07081},
      year={2018}
    }

## Attribution       

- [Spatial Transformer Network](https://github.com/kevinzakka/spatial-transformer-network)
