import { Canvas, createCanvas } from 'canvas';
import fs from 'fs';
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
function pixelArrayToImage(pixels: number[]): Canvas {
  const canvas = createCanvas(28, 28);
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(28, 28);

  let i = 0;
  for (const px of pixels) {
    const gray = Math.round(px * 255);
    imageData.data[i++] = gray; // R
    imageData.data[i++] = gray; // G
    imageData.data[i++] = gray; // B
    imageData.data[i++] = 255; // A
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

function saveAsPng(canvas: Canvas, toFile: string) {
  const buffer = canvas.toBuffer('image/png');
  fs.writeFileSync(toFile, buffer);
}

function initializeDecoder(
  encoderModel: Network<number>,
  encoderModelFilePath: string,
): Decoder<Canvas, number> {
  console.log('Initializing a new decoder...');
  return new Decoder<Canvas, number>(
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
    iterations: 20,
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
  let nn: Decoder<Canvas, number>;
  if (pretrainedModelFilePath != null) {
    if (!fs.existsSync(pretrainedModelFilePath)) {
      throw new Error(`Missing model at '${pretrainedModelFilePath}'`);
    }

    nn = Decoder.deserialize<Canvas, number>(pretrainedModelFilePath);
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
    iterations: 20,
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

(async () => {
  // await trainEncoder('./models/encoder-v2-90%');
  // await testEncoder('./models/encoder-v2-90%');
  trainDecoder('./models/decoder-v1.json', null, './models/encoder-v2-90%.json');
})();
