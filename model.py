import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, add, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_model(vocab_size, max_caption_length, cnn_output_dim=2048):
    """
    Constructs the CNN-LSTM image captioning architecture from the reference notebook:
    - Image Branch: Input(2048) -> BatchNorm -> Dense(256, relu) -> BatchNorm
    - Caption Branch: Input(max_caption_length) -> Embedding(vocab_size, 256, mask_zero=True) -> LSTM(256)
    - Merge: add([image_dense, lstm_out]) -> Dense(256, relu) -> Dense(vocab_size, softmax)
    """
    # 1. Feature input from CNN (InceptionV3)
    input_image = Input(shape=(cnn_output_dim,), name='Features_Input')
    fe1 = BatchNormalization()(input_image)
    fe2 = Dense(256, activation='relu')(fe1)
    fe3 = BatchNormalization()(fe2)

    # 2. Sequence input for tokenized captions
    input_caption = Input(shape=(max_caption_length,), name='Sequence_Input')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(input_caption)
    se2 = LSTM(256)(se1)

    # 3. Decoder / Merge
    decoder1 = add([fe3, se2])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax', name='Output_Layer')(decoder2)

    model = Model(inputs=[input_image, input_caption], outputs=outputs, name='Image_Captioning')
    return model


def compile_caption_model(model, learning_rate=0.01, clipnorm=1.0):
    """Compiles the model with Adam optimizer and categorical cross-entropy loss."""
    optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model
