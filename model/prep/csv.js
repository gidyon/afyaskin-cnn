const tf = require('@tensorflow/tfjs-node');
const meta = require('./../../meta')

async function datasetFromCSV() {
    const csvDataset = tf.data
    .csv(
    meta.CSV_DATASET_FILE_PATH, {
        hasHeader: true,
            columnConfigs: {
                label: { isLabel: true }
            },
        }
    )

    // shuffle dataset
    const tensorsDataset = await csvDataset.shuffle(100).toArray()

    // test dataset
    const testTensorsDataset = tensorsDataset.splice(0, meta.TEST_DATA_NUM)

    // training data
    const trainingDataset = prepareData(tensorsDataset)
    const trainXS = trainingDataset.features
    const trainYS = trainingDataset.labels

    // testing data
    const testDataset = prepareData(testTensorsDataset)
    const testXS = testDataset.features
    const testYS = testDataset.labels

    console.log(`Tensors in memory: ${tf.memory().numTensors}`)
    console.log(`Features shape: ${trainXS.shape}`)
    console.log(`Labels shape: ${trainYS.shape}`)

    return {
        trainXS,
        trainYS,
        testXS,
        testYS,
    }
}


const prepareData = (tensorsDataset) => {
    const pixels = []
    const labels = []

    // Here is where we do cleaning
    tensorsDataset.map(v => {
        pixels.push(Object.values(v.xs).map(v => v/255))
        labels.push(v.ys.label)
    })

    const xs = tf.tensor2d(
        pixels, 
        [pixels.length, meta.IMAGE_HEIGHT*meta.IMAGE_HEIGHT*meta.IMAGE_CHANNELS], 
        "float32",
    )

    const ys = tf.tidy(() => {
        const labelsTensors = tf.tensor1d(labels, "int32")
        return tf.oneHot(labelsTensors, meta.LABELS_LIST.length)
    })

    return {
        features: xs, 
        labels: ys,
    }
}

module.exports = {datasetFromCSV}