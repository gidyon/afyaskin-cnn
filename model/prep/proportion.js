const tf = require('@tensorflow/tfjs-node')
const meta = require('../meta')

async function datasetSumary() {
    // read labels from csv
    const featuresLabel = tf.data
    .csv(
        meta.METADATA_FILE_PATH, {
            hasHeader: true,
            columnConfigs: {
                'dx': {
                    required: true,
                    dtype: 'string',
                    isLabel: false
                },
            },
            configuredColumnsOnly: true,
        }
    )

    // read labels index from csv
    const featuresLabelIndex = tf.data
    .csv(
        meta.CSV_DATASET_FILE_PATH, {
            hasHeader: true,
            columnConfigs: {
                'label': {
                    required: true,
                    dtype: 'int',
                    isLabel: false
                },
            },
            configuredColumnsOnly: true,
        }
    )

    // create labels map
    const labelArray = await featuresLabel.toArray()

    // labels map of indexes
    const labelIndexArray = await featuresLabelIndex.toArray()

    const labelMeta = Object.create(null, {})

    let i = 0
    for (const v of labelArray) {
        labelMeta[v['dx']] = labelIndexArray[i].label   
        i++
    }

    console.log(labelMeta)
}

tf.tidy(() => datasetSumary())