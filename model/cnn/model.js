const tf = require('@tensorflow/tfjs-node');
const meta = require('./../../meta')


// Convolution network with three convolutions, and a classification layer
function createModel() {
    const model = tf.sequential();

    // FIRST LAYER
    // convolution layer
    model.add(tf.layers.conv2d({
        inputShape: [meta.IMAGE_WIDTH, meta.IMAGE_HEIGHT, meta.IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    // Max pooling layer
    model.add(tf.layers.maxPooling2d({poolSize: [3, 3], strides: [2, 2]}));

    // SECOND LAYER
    // convolution layer
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    // Max pooling layer
    model.add(tf.layers.maxPooling2d({poolSize: [3, 3], strides: [2, 2]}));

    // THIRD LAYER;
    // convolution layer
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    // Max pooling layer
    model.add(tf.layers.maxPooling2d({poolSize: [3, 3], strides: [2, 2]}));

    // FLATTEN LAYER
    // We flatten the output from 2D filters into a 1D vector
    model.add(tf.layers.flatten());

    // DENSE LAYERS

    // CLASSIFICATION LAYER
    // Last layer is a dense layer
    model.add(tf.layers.dense({
        units: meta.LABELS_LIST.length,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax',
        // inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, 1],
        // inputDim: 1,
        useBias: true,
    }));

    // optimizer, loss function and metrics
    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy', 'precision'],
    });

    return model
}

module.exports = {createModel}
