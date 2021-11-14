# EpyNN/epynnlive/dummy_time/prepare_dataset.py
# Standard library imports
import random

# Related third party imports
import numpy as np


def features_time(TIME=1, SAMPLING_RATE=128):
    """Generate dummy time features.

    Time features may be white noise or a sum with a pure sine-wave.

    The pure sin-wave has random frequency lower than SAMPLING_RATE // 4.

    :param SAMPLING_RATE: Sampling rate (Hz), defaults to 128.
    :type SAMPLING_RATE: int

    :param TIME: Sampling time (s), defaults to 1.
    :type TIME: int

    :return: Time features of shape (N_FEATURES,).
    :rtype: :class:`numpy.ndarray`

    :return: White noise of shape (N_FEATURES,).
    :rtype: :class:`numpy.ndarray`
    """
    # Number of features describing a sample
    N_FEATURES = TIME * SAMPLING_RATE

    # Initialize features array
    features = np.linspace(0, TIME, N_FEATURES, endpoint=False)

    # Random choice of true signal frequency
    signal_frequency = random.uniform(0, SAMPLING_RATE // 4)

    # Generate pure sine wave of N_FEATURES points
    features = np.sin(2 * np.pi * signal_frequency * features)

    # Generate white noise
    white_noise = np.random.normal(0, scale=0.5, size=N_FEATURES)

    # Random choice between noisy signal or white noise
    features = random.choice([features + white_noise, white_noise])

    return features, white_noise


def label_features(features, white_noise):
    """Prepare label associated with features.

    The dummy law is:

    True signal in features = positive.
    No true signal in features = negative.

    :return: Time features of shape (N_FEATURES,).
    :rtype: :class:`numpy.ndarray`

    :return: White noise of shape (N_FEATURES,).
    :rtype: :class:`numpy.ndarray`

    :return: Single-digit label with respect to features.
    :rtype: int
    """
    # Single-digit positive and negative labels
    p_label = 0
    n_label = 1

    # Test if features contains signal (0)
    if any(features != white_noise):
        label = p_label

    # Test if features is equal to white noise (1)
    elif all(features == white_noise):
        label = n_label

    return label


def prepare_dataset(N_SAMPLES=100):
    """Prepare a set of dummy time sample features and label.

    :param N_SAMPLES: Number of samples to generate, defaults to 100.
    :type N_SAMPLES: int

    :return: Set of sample features.
    :rtype: tuple[:class:`numpy.ndarray`]

    :return: Set of single-digit sample label.
    :rtype: tuple[int]
    """
    # Initialize X and Y datasets
    X_features = []
    Y_label = []

   # Iterate over N_SAMPLES
    for i in range(N_SAMPLES):

        # Compute random time features
        features, white_noise = features_time()

        # Retrieve label associated with features
        label = label_features(features, white_noise)

        # From n measurements to n steps with 1 measurements
        features = np.expand_dims(features, 1)

        # Append sample features to X_features
        X_features.append(features)

        # Append sample label to Y_label
        Y_label.append(label)

    # Prepare X-Y pairwise dataset
    dataset = list(zip(X_features, Y_label))

    # Shuffle dataset
    random.shuffle(dataset)

    # Separate X-Y pairs
    X_features, Y_label = zip(*dataset)

    return X_features, Y_label
