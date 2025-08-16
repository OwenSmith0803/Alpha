import fs from 'fs';
import readline from 'readline';

export async function getMnistData(filename: string): Promise<[number[][], number[][], number[]]> {
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

export async function getMnistDataBatched(
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
