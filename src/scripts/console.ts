import fs from 'fs';
import sharp from 'sharp';
import { Decoder } from '../Decoder';
import { Network, NeuralNetwork } from '../NeuralNetwork';
import { train } from '../train';
import '../utils/arrayUtils';
import { getMnistData, getMnistDataBatched } from './getMnistData';

function outputArrayToClassification(out: number[]) {
  let guess = 0;
  for (let i = 1; i < out.length; i++) {
    if (out[i] > out[guess]) guess = i;
  }
  return guess;
}

function initializeEncoder(): Network<number> {
  console.log('Initializing a new encoder...');
  return new NeuralNetwork<number, number>(
    [784, 64, 64, 10],
    outputArrayToClassification,
    outputArrayToClassification,
    0.01,
    NeuralNetwork.activationFunctions.sigmoid,
    NeuralNetwork.costFunctions.crossEntropy,
    NeuralNetwork.outputLayerActivationFunctions.softmax,
  );
}

// GPT generated
type Image = Buffer<ArrayBufferLike>;
function pixelArrayToImage(pixels: number[]): Image {
  const width = 28;
  const height = 28;
  const rgba = Buffer.alloc(width * height * 4);

  let i = 0;
  for (const px of pixels) {
    const gray = Math.round(px * 255);
    rgba[i++] = gray; // R
    rgba[i++] = gray; // G
    rgba[i++] = gray; // B
    rgba[i++] = gray; // A
  }
  return rgba;
}

// Convert Image type to PNG
async function saveAsPng(rgba: Image, toFile: string) {
  const width = 28;
  const height = 28;

  const pngBuffer = await sharp(rgba, {
    raw: {
      width,
      height,
      channels: 4, // RGBA
    },
  })
    .png()
    .toBuffer();

  fs.writeFileSync(toFile, pngBuffer);
}

function initializeDecoder(
  encoderModel: Network<number>,
  encoderModelFilePath: string,
): Decoder<Image, number> {
  console.log('Initializing a new decoder...');
  return new Decoder<Image, number>(
    [10, 64, 64, 784],
    pixelArrayToImage,
    [encoderModel, encoderModelFilePath],
    0.01,
    NeuralNetwork.activationFunctions.sigmoid,
    // NeuralNetwork.activationFunctions.leakyRelu,
    // NeuralNetwork.outputLayerActivationFunctions.sigmoid,
  );
}

async function trainEncoder(saveToFile: string, pretrainedModelFilePath?: string) {
  const [inputBatches_train, targetBatches_train, labelBatches_train] = await getMnistDataBatched(
    './src/datasets/MNIST_CSV/mnist_train.csv',
    32,
  );
  const [inputData_test, targetData_test, labelData_test] = await getMnistData(
    './src/datasets/MNIST_CSV/mnist_test.csv',
  );

  // initialize/train neural network
  const nn =
    pretrainedModelFilePath && fs.existsSync(pretrainedModelFilePath)
      ? NeuralNetwork.deserialize<number, number>(pretrainedModelFilePath)
      : initializeEncoder();

  train({
    inputBatches_train,
    targetBatches_train,
    labelBatches_train,
    inputData_test,
    targetData_test,
    labelData_test,
    epochs: 20,
    nn,
    saveToFile,
  });
}

async function trainDecoder(saveToFile: string, pretrainedModelFilePath: string): Promise<void>;
async function trainDecoder(
  saveToFile: string,
  pretrainedModelFilePath: null,
  encoderModelFilePath: string,
): Promise<void>;

async function trainDecoder(
  saveToFile: string,
  pretrainedModelFilePath: string | null,
  encoderModelFilePath?: string,
) {
  // Note: we are using the Mnist target data (arrays of length 10) to train/test the decoder model
  const [targetBatches_train, inputBatches_train, labelBatches_train] = await getMnistDataBatched(
    './src/datasets/MNIST_CSV/mnist_train.csv',
    32,
  );
  const [targetData_test, inputData_test, labelData_test] = await getMnistData(
    './src/datasets/MNIST_CSV/mnist_test.csv',
  );

  // initialize/train neural network
  let nn: Decoder<Image, number>;
  if (pretrainedModelFilePath != null) {
    if (!fs.existsSync(pretrainedModelFilePath)) {
      throw new Error(`Missing model at '${pretrainedModelFilePath}'`);
    }

    nn = Decoder.deserialize<Image, number>(pretrainedModelFilePath);
  } else {
    // Deserialize the encoder
    const encoderModel = NeuralNetwork.deserialize<number, number>(encoderModelFilePath!);

    // Initialize the decoder
    nn = initializeDecoder(encoderModel, encoderModelFilePath!);
  }

  train({
    inputBatches_train,
    targetBatches_train,
    labelBatches_train,
    inputData_test,
    targetData_test,
    labelData_test,
    epochs: 20,
    nn,
    saveToFile,
  });
}

async function testEncoder(modelFilePath: string) {
  let nn: Network<number>;
  if (fs.existsSync(modelFilePath)) {
    nn = NeuralNetwork.deserialize<number, number>(modelFilePath);
  } else {
    throw new Error(`Missing model at '${modelFilePath}'`);
  }

  console.log('Testing model...');
  const [inputData_test, targetData_test, labelData_test] = await getMnistData(
    './src/datasets/MNIST_CSV/mnist_test.csv',
  );

  const testingStats = nn.testBatch(inputData_test, targetData_test, labelData_test);
  const testingCost = testingStats.averageCost;
  const testingAccuracy = testingStats.accuracy * 100;
  console.log(`Cost: ${testingCost.toFixed(2)}; Accuracy: ${testingAccuracy.toFixed(2)}%`);
}

async function testDecoder(modelFilePath: string) {
  let nn: Decoder<Image, number>;
  if (fs.existsSync(modelFilePath)) {
    nn = Decoder.deserialize<Image, number>(modelFilePath);
  } else {
    throw new Error(`Missing model at '${modelFilePath}'`);
  }

  console.log('Testing model...');
  // Note: we are using the Mnist target data (arrays of length 10) to train/test the decoder model
  const [targetData_test, inputData_test, labelData_test] = await getMnistData(
    './src/datasets/MNIST_CSV/mnist_test.csv',
  );

  const testingStats = nn.testBatch(inputData_test, targetData_test, labelData_test);
  const testingCost = testingStats.averageCost;
  const testingAccuracy = testingStats.accuracy * 100;
  console.log(`Cost: ${testingCost.toFixed(2)}; Accuracy: ${testingAccuracy.toFixed(2)}%`);
}

async function generateImages(modelFilePath: string) {
  console.log('Generating images...');
  let decoder: Decoder<Image, number>;
  if (fs.existsSync(modelFilePath)) {
    decoder = Decoder.deserialize<Image, number>(modelFilePath);
  } else {
    throw new Error(`Missing model at '${modelFilePath}'`);
  }

  const inputs: number[] = Array(10).fill(0);
  for (let i = 0; i < 10; i++) {
    inputs[i] = 1;
    saveAsPng(decoder.predict(inputs), `./generated-images/generated-${i}.png`);
    inputs[i] = 0;
  }
}

(async () => {
  // await trainEncoder('./models/encoder-v2-90%.json');
  // await testEncoder('./models/encoder-v2-90%.json');
  // await trainDecoder('./models/decoder-v1.json', null, './models/encoder-v2-90%.json');
  await testDecoder('./models/decoder-v1.json');
  // await generateImages('./models/decoder-v1.json');
})();
