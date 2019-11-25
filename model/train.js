const train = require('./cnn/training');
const meta = require('./../meta');

async function trainModel() {
    // load datasets from images or csv
    let dataset
    switch (meta.TRAINING_DATASET_FORMAT) {
        case "image": case "images": case "IMAGE": case "IMAGES":
            console.log('Training on raw images dataset')
            dataset = await require('./prep/image').datasetFromImages()
            break;
        default:
            console.log('Training on csv dataset')
            dataset = await require('./prep/csv').datasetFromCSV()
            break;
    }

    // get model
    const model = require('./cnn/model').createModel()

    // train model
    const results = await train.trainModel(
        model,
        dataset.trainXS.reshape(
            [dataset.trainXS.shape[0], meta.IMAGE_WIDTH, meta.IMAGE_HEIGHT, meta.IMAGE_CHANNELS],
        ),
        dataset.trainYS.reshape(
            [dataset.trainXS.shape[0], meta.LABELS_LIST.length],
        ),
    )

    console.log('Training summary ...')
    console.log(results)

    // save model
    await model.save(meta.SAVED_MODEL_DIR);

    // do some prediction and see
    const prediction = model.predict(dataset.testXS.reshape(
        [dataset.testXS.shape[0], meta.IMAGE_WIDTH, meta.IMAGE_HEIGHT, meta.IMAGE_CHANNELS]
    ))
    prediction.print()
}

trainModel()