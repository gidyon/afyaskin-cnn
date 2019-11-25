async function trainModel(model, features, labels) {
    return model.fit(features, labels, {
        epochs: 10,
        shuffle: true,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                console.log(`EPOCH: ${epoch} LOSS: ${logs.loss}`);
            },
        }
    });
}
    
module.exports = {trainModel}