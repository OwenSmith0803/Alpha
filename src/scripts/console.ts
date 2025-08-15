import fs from 'fs';
import readline from 'readline';
import { NeuralNetwork } from '../NeuralNetwork';
import '../utils/arrayUtils';

const nnModelFile = './models/model.json';

async function getMnistData(filename: string): Promise<[number[][], number[][], number[]]> {
  const stream = fs.createReadStream(filename, {
    encoding: 'utf-8',
  });

  const rl = readline.createInterface({
    input: stream,
    crlfDelay: Infinity,
  });

  const inputData: number[][] = [];
  const targetData: number[][] = [];
  const labelData: number[] = [];

  for await (const line of rl) {
    const [num, ...pixels]: number[] = line.split(',').map((x) => parseInt(x));

    const target = Array(10).fill(0);
    target[num] = 1;
    const inputs = pixels.map((x) => x / 255);

    inputData.push(inputs);
    targetData.push(target);
    labelData.push(num);
  }

  return [inputData, targetData, labelData];
}

async function getMnistDataBatched(
  filename: string,
  batchSize = 100,
): Promise<[number[][][], number[][][], number[][]]> {
  const [inputData, targetData, labelData] = await getMnistData(filename);
  return [
    inputData.chunk(batchSize, false),
    targetData.chunk(batchSize, false),
    labelData.chunk(batchSize, false),
  ];
}

function train({
  inputBatches_train,
  targetBatches_train,
  labelBatches_train,
  inputData_test,
  targetData_test,
  labelData_test,
  iterations,
  pretrainedNetwork,
}: {
  inputBatches_train: number[][][];
  targetBatches_train: number[][][];
  labelBatches_train: number[][];
  inputData_test?: number[][];
  targetData_test?: number[][];
  labelData_test?: number[];
  iterations: number;
  pretrainedNetwork?: NeuralNetwork | undefined;
}): NeuralNetwork {
  function isCorrect(out: number[], label: number) {
    let guess = 0;
    for (let i = 1; i < out.length; i++) {
      if (out[i] > out[guess]) guess = i;
    }

    return guess === label;
  }

  const nn =
    pretrainedNetwork != undefined
      ? pretrainedNetwork
      : new NeuralNetwork(
          [784, 16, 16, 10],
          isCorrect,
          0.05,
          NeuralNetwork.activationFunctions.leakyRelu,
          NeuralNetwork.costFunctions.crossEntropy,
          NeuralNetwork.outputLayerActivationFunctions.softmax,
        );

  // Create the folder if it doesn't exist
  if (!fs.existsSync('./models')) {
    fs.mkdirSync('./models');
  }

  console.log('Training model...');
  const timestamp = Date.now();
  for (let iter = 0; iter < iterations; iter++) {
    console.log(`Iteration: ${iter + 1}/${iterations}`);
    // shuffle the data
    for (let i = inputBatches_train.length - 1; i >= 0; i--) {
      const randInd = Math.floor(Math.random() * (i + 1)); // generate random number between [0, i]
      [inputBatches_train[randInd], inputBatches_train[i]] = [
        inputBatches_train[i],
        inputBatches_train[randInd],
      ];
      [targetBatches_train[randInd], targetBatches_train[i]] = [
        targetBatches_train[i],
        targetBatches_train[randInd],
      ];
      [labelBatches_train[randInd], labelBatches_train[i]] = [
        labelBatches_train[i],
        labelBatches_train[randInd],
      ];
    }

    // train the model
    let totalTrainingIterationCost = 0;
    let totalTrainingIterationAccuracy = 0;
    const batchSize = targetBatches_train.length;
    for (let batch = 0; batch < batchSize; batch++) {
      if ((batch + 1) % 50 === 0 || batch + 1 === batchSize)
        console.log(`Batch: ${batch + 1}/${batchSize}`);

      const { averageCost, accuracy } = nn.trainBatch(
        inputBatches_train[batch],
        targetBatches_train[batch],
        labelBatches_train[batch],
      );
      totalTrainingIterationCost += averageCost;
      totalTrainingIterationAccuracy += accuracy;
    }

    // training stats
    const trainingCost = totalTrainingIterationCost / batchSize;
    const trainingAccuracy = totalTrainingIterationAccuracy / batchSize;
    console.log(
      `Training | Cost: ${trainingCost.toFixed(2)}; Accuracy: ${trainingAccuracy.toFixed(2)}%`,
    );

    // test the model
    if (inputData_test != null && targetData_test != null && labelData_test != null) {
      const testingStats = nn.testBatch(inputData_test, targetData_test, labelData_test);

      // testing stats
      const testingCost = testingStats.averageCost;
      const testingAccuracy = testingStats.accuracy;
      console.log(
        `Testing | Cost: ${testingCost.toFixed(2)}; Accuracy: ${testingAccuracy.toFixed(2)}%`,
      );
    }
    fs.writeFileSync(nnModelFile, nn.serialize());
  }
  console.log(`Training complete in ${(Date.now() - timestamp) / 1000}s`);
  return nn;
}

async function doTrain() {
  const [inputBatches_train, targetBatches_train, labelBatches_train] = await getMnistDataBatched(
    './src/datasets/MNIST_CSV/mnist_train.csv',
    256,
  );
  const [inputData_test, targetData_test, labelData_test] = await getMnistData(
    './src/datasets/MNIST_CSV/mnist_test.csv',
  );

  // initialize/train neural network
  const nn = fs.existsSync(nnModelFile) ? NeuralNetwork.deserialize(nnModelFile) : undefined;
  train({
    inputBatches_train,
    targetBatches_train,
    labelBatches_train,
    inputData_test,
    targetData_test,
    labelData_test,
    iterations: 20,
    pretrainedNetwork: nn,
  });
}

async function testRun() {
  let nn: NeuralNetwork;
  if (fs.existsSync(nnModelFile)) {
    nn = NeuralNetwork.deserialize(nnModelFile);
  } else {
    throw new Error(`Missing model at ${nnModelFile}`);
  }

  console.log('Testing model...');
  const [inputData_test, targetData_test, labelData_test] = await getMnistData(
    './src/datasets/MNIST_CSV/mnist_test.csv',
  );

  const testingStats = nn.testBatch(inputData_test, targetData_test, labelData_test);
  console.log(
    `Cost: ${testingStats.averageCost.toFixed(2)}; Accuracy: ${testingStats.accuracy.toFixed(2)}%`,
  );
}

(async () => {
  // await doTrain();
  await testRun();
})();
