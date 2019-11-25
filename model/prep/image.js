const tf = require('@tensorflow/tfjs-node')
const path = require('path')
const sharp = require('sharp')
const fs = require('fs')
const isImage = require('is-image')

const meta = require('./../../meta')

async function datasetFromImages() {
    try{   
        // read labels from csv
        const featuresMeta = tf.data
        .csv(
            meta.METADATA_FILE_PATH, {
                hasHeader: true,
                columnConfigs: {
                    'lesion_id': {
                        required: true,
                        dtype: 'string',
                        isLabel: false
                    },
                    'image_id': {
                        required: true,
                        dtype: 'string',
                        isLabel: false
                    },
                    'dx': {
                        required: true,
                        dtype: 'string',
                        isLabel: false
                    },
                },
                configuredColumnsOnly: true,
            }
        )

        // create labels map
        const labelMeta = Object.create(null, {})
        const labelArray = await featuresMeta.toArray()
        labelArray.map(column => {
            labelMeta[column['image_id']] = column
        })

        const features = []
        const labels = []

        for (const dir of meta.IMAGES_DIRECTORIES) {
            const files = fs.readdirSync(dir)

            // shuffle the files/features
            shuffle(files)

            for (let file of files) {
                file = path.join(dir, file)
                
                // make sure the file is image
                if (isImage(file) == false) {
                    continue
                }
                
                // resize the input image
                const buf = await sharp(file)
                .resize({
                    width: meta.IMAGE_HEIGHT,
                    height: meta.IMAGE_WIDTH,
                    // fit: sharp.fit.outside,
                })
                .toBuffer()

                const tensor = tf.tidy(() => tf.node.decodeImage(
                    buf, meta.IMAGE_CHANNELS, 'int32', true
                ))

                const arrTensor = tf.tidy(() => tensor.flatten().arraySync())
                tensor.dispose()
                
                // normalize pixel value between 0 - 1
                arrTensor.forEach((element, index) => {
                    arrTensor[index] = element/255    
                });
                
                features.push(arrTensor)
                
                const imageId = path.parse(file).name
                labels.push(labelMeta[imageId])
            }
        }
        
        // split data into training, validation and testin
        // NB: here we split into training and testing as the training algorith will split 
        // the training data into training and validation by specified proportion.
        const testFeatures = features.splice(0, meta.TEST_DATA_NUM)
        const testLabels = labels.splice(0, meta.TEST_DATA_NUM)
        

         // training data
        const trainingDataset = prepareData(features, labels)
        const trainXS = trainingDataset.features
        const trainYS = trainingDataset.labels

        // testing data
        const testDataset = prepareData(testFeatures, testLabels)
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
    } catch(err) {
        console.log('Error has happened')
        console.log(err)
    }
}

function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
      let j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
}

function prepareData(rawFeatures, rawLabels) {
    const xs = tf.tidy(() => {
        return tf.tensor2d(
            rawFeatures,
            [rawFeatures.length, meta.IMAGE_WIDTH*meta.IMAGE_HEIGHT*meta.IMAGE_CHANNELS],
            'float32',
        )
    })

    const ys = tf.tidy(() => {
        const labelsTensor = tf.tensor1d(rawLabels, 'int32')
        return tf.oneHot(labelsTensor, meta.LABELS_LIST.length)
    })

    return {
        features: xs, 
        labels: ys,
    }
}

const numPixels = meta.IMAGE_WIDTH*meta.IMAGE_HEIGHT*meta.IMAGE_CHANNELS

function createTensor(rawFeatures) {
    return tf.tidy(() => {
        return tf.tensor2d(rawFeatures, [rawFeatures.length, numPixels], 'float32')
    })
}

async function convertImageToTensor(image) {
    // resize the input image
    const buf = await sharp(image)
    .resize({
        width: meta.IMAGE_HEIGHT,
        height: meta.IMAGE_WIDTH,
        // fit: sharp.fit.outside,
    })
    .toBuffer()
    
    const tensor = tf.tidy(() => tf.node.decodeImage(
        buf, meta.IMAGE_CHANNELS, 'int32', true),
    )
    const arrTensor = tf.tidy(() => tensor.flatten().arraySync())
    
    // normalize
    arrTensor.forEach((element, index) => {
        arrTensor[index] = element/255    
    });

    return arrTensor
}

module.exports = {datasetFromImages, convertImageToTensor, createTensor}