const CSV_DATASET_FILE_PATH = process.env.CSV_DATASET_FILE_PATH || 
	"file:///home/meshack/Desktop/afyacnn/datasets/hmnist_28_28_RGB.csv"

const METADATA_FILE_PATH = process.env.METADATA_FILE_PATH ||
	"file:///home/meshack/Desktop/afyacnn/datasets/HAM10000_metadata.csv"

const SAVED_MODEL_DIR = process.env.SAVED_MODEL_PATH ||
	"file:///home/meshack/Desktop/afyacnn/model/saved"

const SAVED_MODEL_PATH = process.env.SAVED_MODEL_PATH ||
		"file:///home/meshack/Desktop/afyacnn/model/saved/model.json"

const CLASSIFICATION_THRESHOLD = process.env.CLASSIFICATION_THRESHOLD || 0.5

const TRAINING_DATASET_FORMAT = process.env.TRAINING_DATASET_FORMAT || 'csv'

const _ = (v) => (v != '' && v != undefined) ? v.split(',') : undefined

const IMAGES_DIRECTORIES = _(process.env.IMAGES_DIRECTORIES) ||
	[
		"/home/meshack/Desktop/afyacnn/datasets/1", 
		"/home/meshack/Desktop/afyacnn/datasets/2",
	]

const IMAGES_OUTPUT_DIRECTORY = process.env.IMAGES_OUTPUT_DIRECTORY ||
	"/home/meshack/Desktop/afyacnn/output"

const IMAGE_HEIGHT = process.env.IMAGE_HEIGHT || 28
const IMAGE_WIDTH = process.env.IMAGE_WIDTH || 28
const IMAGE_CHANNELS = process.env.IMAGE_CHANNELS || 3

const LABELS_LIST = process.env.LABEL_LIST || [0, 1, 2, 3, 4, 5, 6]

const TEST_DATA_NUM = process.env.TEST_DATA_NUM || 5

module.exports = {
	CSV_DATASET_FILE_PATH,
	METADATA_FILE_PATH,
	IMAGES_DIRECTORIES,
	IMAGES_OUTPUT_DIRECTORY,
	IMAGE_HEIGHT,
	IMAGE_WIDTH,
	IMAGE_CHANNELS,
	LABELS_LIST,
	SAVED_MODEL_DIR,
	SAVED_MODEL_PATH,
	CLASSIFICATION_THRESHOLD,
	TRAINING_DATASET_FORMAT,
	TEST_DATA_NUM,
}