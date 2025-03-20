import tensorflow as tf

def dataset_generator(X, Y, labels, stims, samp_types):
    """
    Generator function for creating a TensorFlow dataset.

    :param X: Input features
    :param Y: Additional input features
    :param labels: Labels for prediction
    :param stims: Stimulus-related labels
    :param samp_types: Sample type labels
    :yield: Tuple of (inputs, outputs) for model training
    """
    num_samples = X.shape[0]
    
    for i in range(num_samples):
        yield (X[i], Y[i]), (
            X[i],  # X_hat (Reconstruction of gene expression X)
            tf.zeros((1,), dtype=tf.float32),  # (Latent space placeholder)
            tf.zeros((1,), dtype=tf.float32),  # (Latent space placeholder)
            tf.zeros((1,), dtype=tf.float32),  # (Latent space placeholder)
            tf.zeros((1,), dtype=tf.float32),  # (Latent space placeholder)
            labels[i],  # ground truth for sample_id/label prediction
            stims[i],  # ground truth for stim prediction
            samp_types[i],  # ground truth for samp_type prediction
            Y[i]  #  ground truth for Y (cell type proportion)
        )

def dataset_generator_unsupervised(X, labels, stims, samp_types):
    """
    Generator function for creating a TensorFlow dataset for buddi unsupervised training.

    :param X: Input features
    :param labels: Labels for prediction
    :param stims: Stimulus-related labels
    :param samp_types: Sample type labels
    :yield: Tuple of (inputs, outputs) for model training
    """
    num_samples = X.shape[0]
    
    for i in range(num_samples):
        yield (X[i]), (
            X[i],  # X_hat (Reconstruction of gene expression X)
            tf.zeros((1,), dtype=tf.float32),  # (Latent space placeholder)
            tf.zeros((1,), dtype=tf.float32),  # (Latent space placeholder)
            tf.zeros((1,), dtype=tf.float32),  # (Latent space placeholder)
            tf.zeros((1,), dtype=tf.float32),  # (Latent space placeholder)
            labels[i],  # ground truth for sample_id/label prediction
            stims[i],  # ground truth for stim prediction
            samp_types[i],  # ground truth for samp_type prediction
            tf.zeros((1,), dtype=tf.float32) # dummy placeholder for Y as it is not available
        )

def get_output_signature(
    X, Y, labels, stims, samp_types
    ):

    output_signature = (
        (tf.TensorSpec(shape=X.shape[1:], dtype=tf.float32), tf.TensorSpec(shape=Y.shape[1:], dtype=tf.float32)),  # Inputs
        (
            tf.TensorSpec(shape=X.shape[1:], dtype=tf.float32),  # Ground truth for X
            tf.TensorSpec(shape=(1,), dtype=tf.float32),  # (Latent space placeholder)
            tf.TensorSpec(shape=(1,), dtype=tf.float32),  # (Latent space placeholder)
            tf.TensorSpec(shape=(1,), dtype=tf.float32),  # (Latent space placeholder)
            tf.TensorSpec(shape=(1,), dtype=tf.float32),  # (Latent space placeholder)
            tf.TensorSpec(shape=labels.shape[1:], dtype=tf.int32),  # ground truth for sample_id/label prediction
            tf.TensorSpec(shape=stims.shape[1:], dtype=tf.int32),  # ground truth for stim prediction
            tf.TensorSpec(shape=samp_types.shape[1:], dtype=tf.int32),  # ground truth for samp_type prediction
            tf.TensorSpec(shape=Y.shape[1:], dtype=tf.float32)  # Ground truth for Y
        )
    )

    return output_signature

def get_output_signature_unsupervised(
    X, labels, stims, samp_types
    ):

    output_signature = (
        (tf.TensorSpec(shape=X.shape[1:], dtype=tf.float32)),  # Inputs
        (
            tf.TensorSpec(shape=X.shape[1:], dtype=tf.float32),  # Ground truth for X
            tf.TensorSpec(shape=(1,), dtype=tf.float32),  # (Latent space placeholder)
            tf.TensorSpec(shape=(1,), dtype=tf.float32),  # (Latent space placeholder)
            tf.TensorSpec(shape=(1,), dtype=tf.float32),  # (Latent space placeholder)
            tf.TensorSpec(shape=(1,), dtype=tf.float32),  # (Latent space placeholder)
            tf.TensorSpec(shape=labels.shape[1:], dtype=tf.int32),  # ground truth for sample_id/label prediction
            tf.TensorSpec(shape=stims.shape[1:], dtype=tf.int32),  # ground truth for stim prediction
            tf.TensorSpec(shape=samp_types.shape[1:], dtype=tf.int32),  # ground truth for samp_type prediction
            tf.TensorSpec(shape=(1,), dtype=tf.float32)  # Dummy placeholder for Y
        )
    )

    return output_signature