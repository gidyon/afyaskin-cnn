const tf = require('@tensorflow/tfjs-node')
const express = require('express')
const bodyParser = require('body-parser')
const multer = require('multer')
const meta = require('./meta')
const {createTensor, convertImageToTensor} = require('./model/prep/image')
const {doPrediction} = require('./model/predict')

const app = express()

app.use(bodyParser.json())
app.use(bodyParser.urlencoded({extended: true}))    

let upload = multer({
    limits: {
        fileSize: 4 * 1024 * 1024,
    }    
})

async function start() {
    const model = await tf.loadLayersModel(meta.SAVED_MODEL_PATH)
    const threshold = 0.5

    const classify = async (req, res) => {
        try {
            // files must be provided
            if (req.files == undefined || req.files.length == 0) {
                // bad request
                res.status(400).json({error: 'Please provide image(s) to scan'})
                return
            }

            // hold features
            const imagesTensors = []

            for (const file of req.files) {
                const imageTensor = await convertImageToTensor(file.buffer)
                imagesTensors.push(imageTensor)
            }

            const tensors = createTensor(imagesTensors)

            // we predict on multiple images at once
            const predictions = await doPrediction(
                model, tensors, meta.CLASSIFICATION_THRESHOLD || threshold,
            )

            // add extra information to predictions
            let i = 0
            for (const file of req.files) {
                predictions[i]['originalFileName'] = file.originalname
                predictions[i]['fileName'] = file.filename,
                predictions[i]['fileSize'] = file.size,
                i++
            }

            // send the predictions to the client
            res.status(200).json({predictions: predictions})
        } catch (err) {
            res.status(401).json({error: err})
        } finally {}
    }

    const process = async () => await classify()

    app.post('/api/classify/', upload.array('images', 6), async function (req, res) {
        tf.tidy(() => {
            process()
        })
    })

    const port = process.env.PORT || 3000
    
    app.listen(port, () => {
        console.log(`Convolutional Neural Net is running on PORT ${port}`)
    })
}

start()