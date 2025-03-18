from typing import Callable, Any

import tensorflow as tf
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error, categorical_crossentropy
import tensorflow.keras.backend as K

LossFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
LossAggregationFn = Callable[[tf.Tensor, Any], tf.Tensor]

def kl_loss_generator(beta: float, 
                      agg_fn: LossAggregationFn = K.sum,
                      **kwargs: Any
    ) -> Callable:
    """KL divergence loss function generator.
    
    :param beta: Weight of the KL divergence loss
    :param agg_fn: Aggregation function for the KL divergence loss
    :param kwargs: Additional arguments to pass to the aggregation function
    :return kl_loss: KL divergence loss function
    """

    def kl_loss(
            y_true: tf.Tensor, 
            y_pred: tf.Tensor
            ) -> tf.Tensor:
        """KL divergence loss function.
        
        :param y_true: Not used. Kept for compatibility with Keras API.
        :param y_pred: Predicted values. 
            In this case this would be the output of a VAE encoder
            which is a concatenation of its mu and log_var output layers.   
        :return kl: KL divergence loss
        """

        # parse the predicted values into mu and log_var
        n_z_dim = tf.shape(y_pred)[-1] // 2
        z_mu, z_log_var = y_pred[:, :n_z_dim], y_pred[:, n_z_dim:]

        kl = -0.5 * agg_fn(1 + z_log_var - K.square(z_mu) - K.exp(z_log_var), **kwargs)
        return beta * kl
    
    return kl_loss

def reconstr_loss_generator(
        weight: float = 1.0,
        agg_fn: LossAggregationFn = K.sum,
        **kwargs: Any
    ) -> Callable:
    """Reconstruction loss function generator.
    
    :param weight: Weight of the reconstruction loss
    :param agg_fn: Aggregation function for the reconstruction loss
    :param kwargs: Additional arguments to pass to the aggregation function
    :return reconstr_loss: Reconstruction loss function
    """

    def reconstr_loss(
            y_true: tf.Tensor, 
            y_pred: tf.Tensor
            ) -> tf.Tensor:
        """
        Reconstruction loss function.
        
        :param y_true: True values
        :param y_pred: Predicted values
        :return reconstr_loss: Reconstruction loss
        """

        reconstr_loss = mean_squared_error(y_true, y_pred)
        return weight * agg_fn(reconstr_loss, **kwargs)
    
    return reconstr_loss

def classifier_loss_generator(
        weight: float = 1.0,
        loss_fn: LossFn = categorical_crossentropy, # or mean_absolute_error
        agg_fn: LossAggregationFn = K.sum,
        **kwargs: Any
    ) -> Callable:
    """Classifier loss function generator.
    
    :param weight: Weight of the classifier loss
    :param agg_fn: Aggregation function for the classifier loss
    :param kwargs: Additional arguments to pass to the aggregation function
    :return classifier_loss: Classifier loss function
    """

    def classifier_loss(
            y_true: tf.Tensor, 
            y_pred: tf.Tensor
            ) -> tf.Tensor:
        """Cross-entropy loss function for classifier.
        
        :param y_true: True values
        :param y_pred: Predicted values
        :return class_loss: Cross-entropy loss
        """

        class_loss = loss_fn(y_true, y_pred)
        return weight * agg_fn(class_loss, **kwargs)
    
    return classifier_loss

def unsupervised_dummy_loss_fn(
        y_true: tf.Tensor, 
        y_pred: tf.Tensor
        ) -> tf.Tensor:
    """
    Dummy loss function for unsupervised branch proportion estimator.
    
    :param y_true: True values. Not used.
    :param y_pred: Predicted values. Not used.
    :return dummy_loss: Zero loss
    """
    return tf.constant(0.0)