import fs from 'fs';
import { NeuralNetwork } from './NeuralNetwork';

export function train<T, U>({
  inputBatches_train,
  targetBatches_train,
  labelBatches_train,
  inputData_test,
  targetData_test,
  labelData_test,
  epochs,
  nn,
  saveToFile,
}: {
  inputBatches_train: number[][][];
  targetBatches_train: number[][][];
  labelBatches_train: U[][];
  inputData_test?: number[][];
  targetData_test?: number[][];
  labelData_test?: U[];
  epochs: number;
  nn: NeuralNetwork<T, U>;
  saveToFile: string;
}): NeuralNetwork<T, U> {
  // create the folder if it doesn't exist
  if (!fs.existsSync('./models')) {
    fs.mkdirSync('./models');
  }

  console.log('Training model...');
  const timestamp = Date.now();
  for (let iter = 0; iter < epochs; iter++) {
    console.log(`Epoch: ${iter + 1}/${epochs}`);
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
    let totalTrainingEpochCost = 0;
    let totalTrainingEpochAccuracy = 0;
    const batchSize = targetBatches_train.length;
    for (let batch = 0; batch < batchSize; batch++) {
      if ((batch + 1) % 50 === 0 || batch + 1 === batchSize)
        console.log(`Batch: ${batch + 1}/${batchSize}`);

      const { averageCost, accuracy } = nn.trainBatch(
        inputBatches_train[batch],
        targetBatches_train[batch],
        labelBatches_train[batch],
      );
      totalTrainingEpochCost += averageCost;
      totalTrainingEpochAccuracy += accuracy;
    }

    // training stats
    const trainingCost = totalTrainingEpochCost / batchSize;
    const trainingAccuracy = (totalTrainingEpochAccuracy / batchSize) * 100;
    console.log(
      `Training | Cost: ${trainingCost.toFixed(2)}; Accuracy: ${trainingAccuracy.toFixed(2)}%`,
    );

    // test the model
    if (inputData_test != null && targetData_test != null && labelData_test != null) {
      const testingStats = nn.testBatch(inputData_test, targetData_test, labelData_test);

      // testing stats
      const testingCost = testingStats.averageCost;
      const testingAccuracy = testingStats.accuracy * 100;
      console.log(
        `Testing | Cost: ${testingCost.toFixed(2)}; Accuracy: ${testingAccuracy.toFixed(2)}%`,
      );
    }
    fs.writeFileSync(saveToFile, nn.serialize());
  }
  console.log(`Training complete in ${(Date.now() - timestamp) / 1000}s`);
  return nn;
}
