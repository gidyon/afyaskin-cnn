const meta = require('../meta')

async function doPrediction(model, features, threshold) {

    const prediction = model.predict(
        features.reshape(
            [features.shape[0], meta.IMAGE_WIDTH, meta.IMAGE_HEIGHT, meta.IMAGE_CHANNELS],
        ),
    )

    const arrPreds = prediction.arraySync()
    const maxPreds = prediction.argMax(1).arraySync()

    const predictions = []

    let i = 0
    for (const index of maxPreds) {
        const label = labels[index]
        const labelText = labelsText[label]
        // classification higher than threshold
        if (arrPreds[i][index] > threshold) {
            predictions.push({
                aboveThreshold: true,
                probability: arrPreds[i][index],
                threshold: threshold,
                labelIndex: index,
                label: label,
                labelText: labelText,
            })
        }else{
            // Not higher than threshold
            predictions.push({
                aboveThreshold: false,
                probability: arrPreds[i][index],
                threshold: threshold,
                labelIndex: index,
                label: label,
                labelText: labelText,
            })
        }
        i++
    }

    return predictions
}

const labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

const labelsText = {
    ['akiec']: 'AKIEC',
    ['bcc']: 'Basal Cell Carcinoma',
    ['bkl']: 'Benign Keratosis',
    ['df']: 'Dermatofibroma',
    ['nv']: 'Melanocytic Nevi',
    ['vasc']: 'Vascular Skin Lesions',
    ['mel']: 'Melanoma',
}

module.exports = {doPrediction}