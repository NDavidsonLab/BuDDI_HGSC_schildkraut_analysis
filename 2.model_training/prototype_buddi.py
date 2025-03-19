from typing import List, Tuple, Dict, Any, Union, Callable

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, ReLU, Concatenate
from tensorflow.keras.optimizers import Adam

from prototype_buddi_components import *
from prototype_buddi_layers import *
from prototype_buddi_losses import *

ActivationFn = Union[str, Callable[[tf.Tensor], tf.Tensor]]

def build_buddi(
        n_x: int,
        n_y: int,
        n_labels: int,
        n_stims: int,
        n_samp_types: int,
        z_dim: int = 64,
        encoder_hidden_dim: int = 512,
        decoder_hidden_dim: int = 512,
        alpha_label: float = 100.0,
        alpha_stim: float = 100.0,
        alpha_samp_type: float = 100.0,
        alpha_prop: float = 100.0,
        beta_kl_slack: float = 0.1,
        beta_kl_label: float = 100.0,
        beta_kl_stim: float = 100.0,
        beta_kl_samp_type: float = 100.0,
        activation: ActivationFn = 'relu',
        optimizer = Adam(learning_rate=0.0005), 
    ) -> Tuple[Model, Model]:
    """
    Builds the BUDDI model.

    :param n_x: Number of features in the input data
    :param n_y: Number of features in the output data
    :param n_labels: Number of unique labels in the data
    :param n_stims: Number of unique stimulation conditions in the data
    :param n_samp_types: Number of unique sample types in the data
    :param z_dim: Dimension of the latent space
    :param encoder_hidden_dim: Dimension of the hidden layers in the encoder
    :param decoder_hidden_dim: Dimension of the hidden layers in the decoder
    :param alpha_label: Weight of the classifier loss for the label branch
    :param alpha_stim: Weight of the classifier loss for the stimulation branch
    :param alpha_samp_type: Weight of the classifier loss for the sample type branch
    :param alpha_prop: Weight of the classifier loss for the proportion estimator branch
    :param beta_kl_slack: Weight of the KL divergence loss for the slack branch
    :param beta_kl_label: Weight of the KL divergence loss for the label branch
    :param beta_kl_stim: Weight of the KL divergence loss for the stimulation branch
    :param beta_kl_samp_type: Weight of the KL divergence loss for the sample type branch
    :param activation: Activation function for the hidden layers
    :param optimizer: Optimizer for the model
    :return supervised_buddi: Supervised BUDDI model
    :return unsupervised_buddi: Unsupervised BUDDI model
    """

    X = Input(shape=(n_x,), name='X')
    Y = Input(shape=(n_y,), name='Y')

    # --------------------- Encoders ---------------------
    ## This encoder branch captures the variation in data that is
    ## explained by sample labels
    encoder_branch_label = build_encoder_branch(
        inputs = X,
        hidden_dim = encoder_hidden_dim,
        z_dim = z_dim,
        activation = activation,
        representation_name = 'label'
    )

    ## This encoder branch captures the variation in data that is
    ## explained by stimulation conditions
    encoder_branch_stim = build_encoder_branch(
        inputs = X,
        hidden_dim = encoder_hidden_dim,
        z_dim = z_dim,
        activation = activation,
        representation_name = 'stim'
    )

    ## This encoder branch captures the variation in data that is
    ## explained by sample types
    encoder_branch_samp_type = build_encoder_branch(
        inputs = X,
        hidden_dim = encoder_hidden_dim,
        z_dim = z_dim,
        activation = activation,
        representation_name = 'samp_type'
    )

    ## This encoder branch absorbs the additional variation in data 
    ## that is not captured by other branches
    encoder_branch_slack = build_encoder_branch(
        inputs = X,
        hidden_dim = encoder_hidden_dim,
        z_dim = z_dim,
        activation = activation,
        representation_name = 'slack'
    )

    ## Each of these outputs are z_mu and z_log_var concatenated
    z_params_label = encoder_branch_label(X)
    z_params_stim = encoder_branch_stim(X)
    z_params_samp_type = encoder_branch_samp_type(X)
    z_params_slack = encoder_branch_slack(X)

    # --------------------- Sampling ---------------------
    z_label = ReparameterizationLayer(name='z_label')(z_params_label)
    z_stim = ReparameterizationLayer(name='z_stim')(z_params_stim)
    z_samp_type = ReparameterizationLayer(name='z_samp_type')(z_params_samp_type)
    z_slack = ReparameterizationLayer(name='z_slack')(z_params_slack)

    # --------------------- Classifier ---------------------
    ## This classifier network predicts sample labels from the latent space
    classifier_branch_label = build_latent_space_classifier(
        inputs = z_label,
        num_classes = n_labels,
        representation_name = 'label'
    )

    ## This classifier network predicts stimulation conditions from the latent space
    classifier_branch_stim = build_latent_space_classifier(
        inputs = z_stim,
        num_classes = n_stims,
        representation_name = 'stim'
    )

    ## This classifier network predicts sample types from the latent space
    classifier_branch_samp_type = build_latent_space_classifier(
        inputs = z_samp_type,
        num_classes = n_samp_types,
        representation_name = 'samp_type'
    )

    ## These are the predicted labels from the latent space
    pred_label = classifier_branch_label(z_label)
    pred_stim = classifier_branch_stim(z_stim)
    pred_samp_type = classifier_branch_samp_type(z_samp_type)

    ## The Proportion Estimator branch predicts cell type proportions from X
    prop_estimator = build_prop_estimator(
        inputs = X,
        num_classes = n_y,
        activation = activation,
        estimator_name = 'prop_estimator'
    )

    Y_hat = prop_estimator(X)

    # --------------------- Decoders ---------------------

    ## Supervised Decoder Input
    supervised_decoder_input = Concatenate(name='supervised_decoder_input')(
        [Y, z_label, z_stim, z_samp_type, z_slack])
    
    unsupervised_decoder_input = Concatenate(name='unsupervised_decoder_input')(
        [Y_hat, z_slack, z_stim, z_samp_type, z_slack])
    
    supervised_decoder, unsupervised_decoder = build_semi_supervised_decoder(
        inputs_supervised = supervised_decoder_input,
        inputs_unsupervised = unsupervised_decoder_input,
        output_dim = n_x,
        hidden_dims = decoder_hidden_dim,
        activation = activation,
        output_activation = 'sigmoid', # Assuming MinMaxNormalized data
        output_name = 'X'
    )

    X_hat_supervised = supervised_decoder(supervised_decoder_input)
    X_hat_unsupervised = unsupervised_decoder(unsupervised_decoder_input)

    # --------------------- Losses ---------------------
    ## Note these are all functions with standard tensorflow signature accepting y_true and y_pred
    kl_loss_fn_label = kl_loss_generator(beta=beta_kl_label, agg_fn=K.sum, axis=-1)
    kl_loss_fn_stim = kl_loss_generator(beta=beta_kl_stim, agg_fn=K.sum, axis=-1)
    kl_loss_fn_samp_type = kl_loss_generator(beta=beta_kl_samp_type, agg_fn=K.sum, axis=-1)
    kl_loss_fn_slack = kl_loss_generator(beta=beta_kl_slack, agg_fn=K.sum, axis=-1)

    reconstr_loss_fn = reconstr_loss_generator(weight=1.0, agg_fn=K.sum, axis=-1)

    classifier_loss_fn_label = classifier_loss_generator(
        loss_fn=mean_absolute_error,
        weight=alpha_label, agg_fn=K.sum, axis=-1)
    classifier_loss_fn_stim = classifier_loss_generator(
        loss_fn=mean_absolute_error,
        weight=alpha_stim, agg_fn=K.sum, axis=-1)
    classifier_loss_fn_samp_type = classifier_loss_generator(
        loss_fn=mean_absolute_error,
        weight=alpha_samp_type, agg_fn=K.sum, axis=-1)
    # this is the loss function for the proportion estimator
    prop_estimator_loss_fn = classifier_loss_generator(
        loss_fn=mean_absolute_error,
        weight=alpha_prop, agg_fn=K.sum, axis=-1)

    # --------------------- Compile ---------------------

    ## Shared output and loss components
    shared_buddi_z_params = [
        z_params_label, z_params_stim, z_params_samp_type, z_params_slack
    ]
    shared_buddi_kl_losses = [
        kl_loss_fn_label, kl_loss_fn_stim, kl_loss_fn_samp_type, kl_loss_fn_slack
    ]

    shared_buddi_pred_outputs= [
        pred_label, pred_stim, pred_samp_type, 
        # note that Y_hat is predicted regardless
        Y_hat 
    ]
    shared_buddi_pred_losses = [
        classifier_loss_fn_label, classifier_loss_fn_stim, classifier_loss_fn_samp_type
        # note Y_hat loss is not included as a shared loss, that is because
        # in the unsupervised branch we don't have the ground truth to train it    
    ]

    supervised_buddi_inputs = [X, Y]
    supervised_buddi_outputs = [X_hat_supervised] + shared_buddi_z_params + shared_buddi_pred_outputs    
    supervisied_buddi_pred_losses = shared_buddi_pred_losses + [prop_estimator_loss_fn]
    supervised_buddi_losses = [
        # standard reconstruction loss
        # corresponding to [X_hat_supervised]
        # note this is shared with the unsupervised branch
        reconstr_loss_fn, 
        # shared KL losses to the latent spaces
        # corresponding to shared_buddi_z_params
        *shared_buddi_kl_losses, 
        # prediction losses specific to supervised branches 
        # corresponding to shared_buddi_pred_outputs
        *supervisied_buddi_pred_losses 
    ]
    
    unsupervised_buddi_inputs = [X]
    unsupervised_buddi_ouputs = [X_hat_unsupervised] + shared_buddi_z_params + shared_buddi_pred_outputs    
    unsupervised_buddi_pred_losses = shared_buddi_pred_losses + [unsupervised_dummy_loss_fn] # dummy loss item here that returns 0
    unsupervised_buddi_losses = [
        # standard reconstruction loss
        # corresponding to [X_hat_unsupervised]
        # note this is shared with the supervised branch
        reconstr_loss_fn,
        # shared KL losses to the latent spaces 
        # corresponding to shared_buddi_z_params
        *shared_buddi_kl_losses, 
        # prediction losses specific to unsupervised branches 
        # corresponding to shared_buddi_pred_outputs
        *unsupervised_buddi_pred_losses 
    ]

    supervised_buddi = Model(
        inputs = supervised_buddi_inputs,
        outputs = supervised_buddi_outputs,
        name = 'supervised_buddi'
    )
    supervised_buddi.compile(optimizer=optimizer, loss=supervised_buddi_losses)

    unsupervised_buddi = Model(
        inputs = unsupervised_buddi_inputs,
        outputs = unsupervised_buddi_ouputs,
        name = 'unsupervised_buddi'
    )
    unsupervised_buddi.compile(optimizer=optimizer, loss=unsupervised_buddi_losses)

    return supervised_buddi, unsupervised_buddi