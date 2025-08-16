import fs from 'fs';
import z from 'zod';
import { Matrix } from './Matrix';
import { MappingFn, trySync } from './utils/utils';

export type CostFunction = (target: number, actual: number) => number;
export type OutputLayerActivationFunction = (out: number[]) => number[];
export type OutputLayerActivationFunctionDeriv = (
  out: number[],
  prenormalized: number[],
  i: number,
  j: number,
) => number;

// T is the outputted type from the .predict() method. U is the label type
// Since T and U are usually the same, this type annotation can make code more concise
export type Network<T> = NeuralNetwork<T, T>;

export class NeuralNetwork<T, U> {
  // first index represents the function; the second index represents the function's derivative
  static activationFunctions: Record<string, [MappingFn, MappingFn]> = {
    relu: [(x) => (x >= 0 ? x : 0), (x) => (x >= 0 ? 1 : 0)],
    leakyRelu: [(x) => (x >= 0 ? x : 0.01 * x), (x) => (x >= 0 ? 1 : 0.01)],
    sigmoid: [
      (x) => 1 / (1 + Math.exp(-x)),
      (x) => {
        const sigmoid = 1 / (1 + Math.exp(-x));
        return sigmoid * (1 - sigmoid);
      },
    ],
    tanh: [Math.tanh, (x) => 1 - Math.pow(Math.tanh(x), 2)],
  };

  static outputLayerActivationFunctions: Record<
    string,
    [OutputLayerActivationFunction, OutputLayerActivationFunctionDeriv]
  > = {
    softmax: [
      (out: number[]) => {
        const sum = out.reduce((acc, val) => acc + Math.exp(val), 0);
        return out.map((val) => Math.exp(val) / sum);
      },
      (out: number[], _: number[], i: number, j: number) => {
        return i === j ? out[i] * (1 - out[i]) : -out[i] * out[j];
      },
    ],
  };

  static costFunctions: Record<string, [CostFunction, CostFunction]> = {
    meanSquaredError: [
      (target: number, actual: number) => Math.pow(target - actual, 2),
      (target: number, actual: number) => 2 * (actual - target),
    ],
    // note: actual is always >0 because it is relayed from the output of the softmax function
    crossEntropy: [
      (target: number, actual: number) => -target * Math.log(actual),
      (target: number, actual: number) => -target / actual,
    ],
  };

  protected structure: number[] = [];
  protected weights: Matrix[] = [];
  protected biases: Matrix[] = [];
  protected activationFn: MappingFn;
  protected derivActivationFn: MappingFn;
  protected costFunction: CostFunction;
  protected derivCostFunction: CostFunction;
  protected outputLayerActivationFn: OutputLayerActivationFunction | null;
  protected derivOutputLayerActivationFn: OutputLayerActivationFunctionDeriv | null;
  constructor(
    structure: number[],
    protected outputMappingFn: (out: number[]) => T,
    protected isCorrectFn: (out: number[]) => U,
    protected learningRate = 0.01,
    [activationFn, derivActivationFn]: [MappingFn, MappingFn] = NeuralNetwork.activationFunctions
      .relu,
    [costFunction, derivCostFunction]: [CostFunction, CostFunction] = NeuralNetwork.costFunctions
      .meanSquaredError,
    [outputLayerActivationFn, derivOutputLayerActivationFn]:
      | [OutputLayerActivationFunction, OutputLayerActivationFunctionDeriv]
      | [null, null] = [null, null],
  ) {
    // configure structure
    this.structure = structure;
    for (let i = 1; i < structure.length; i++) {
      this.weights.push(new Matrix(this.structure[i], this.structure[i - 1], Matrix.initRandom));
      // this.biases.push(new Matrix(this.structure[i], 1, Matrix.initRandom));
      this.biases.push(new Matrix(this.structure[i], 1)); // initialize weights to 0 initially, not random
    }

    // initialize activation/cost functions
    this.activationFn = activationFn;
    this.derivActivationFn = derivActivationFn;
    this.costFunction = costFunction;
    this.derivCostFunction = derivCostFunction;
    this.outputLayerActivationFn = outputLayerActivationFn;
    this.derivOutputLayerActivationFn = derivOutputLayerActivationFn;
  }

  predict(firstArg: number[] | number, ...rest: number[]): T {
    const inputs: number[] = Array.isArray(firstArg) ? firstArg : [firstArg, ...rest];
    const out = this.feedForward(inputs);
    return this.outputMappingFn(out);
  }

  protected feedForward(inputs: number[]): number[] {
    if (inputs.length !== this.structure[0]) {
      throw new Error(`Model expected [${this.structure[0]}] arguments; got [${inputs.length}]`);
    }

    let current: Matrix = Matrix.from1dArray(inputs);
    for (let layer = 0; layer < this.weights.length; layer++) {
      current = Matrix.multiply(this.weights[layer], current).add(this.biases[layer]);

      // only apply activation function if it isn't the last layer or there isn't an output layer activation function
      if (layer < this.weights.length - 1 || this.outputLayerActivationFn == null)
        current = current.map(this.activationFn);
    }

    const prenormalizedOut: number[] = current.to1dArray(); // matrix (that is actually a vector) to number[]
    return this.outputLayerActivationFn == null
      ? prenormalizedOut
      : this.outputLayerActivationFn(prenormalizedOut);
  }

  /**
   * Trains the neural network off of a batch of samples
   *
   * @param inputs the input values, indexed by sample
   * @param target the target values, indexed by sample
   * @returns the average cost of the batch
   */
  trainBatch(
    inputs: number[][],
    target: number[][],
    labels: U[],
  ): { averageCost: number; accuracy: number } {
    if (inputs.length !== target.length) {
      throw new Error(
        `Number of input samples [${inputs.length}] does not match number of target samples [${target.length}]`,
      );
    }
    const batchSize = inputs.length;

    let totalBatchCost = 0;
    let correct = 0;
    const totalDerivWeights: Matrix[] = [];
    const totalDerivBiases: Matrix[] = [];
    for (let i = 1; i < this.structure.length; i++) {
      totalDerivWeights.push(new Matrix(this.structure[i], this.structure[i - 1]));
      totalDerivBiases.push(new Matrix(this.structure[i], 1));
    }

    for (let sample = 0; sample < batchSize; sample++) {
      const { cost, isCorrect, derivWeights, derivBiases } = this.trainOne(
        inputs[sample],
        target[sample],
        labels[sample],
        false,
      );
      totalBatchCost += cost;
      if (isCorrect) correct++;

      for (let layer = 0; layer < this.weights.length; layer++) {
        totalDerivWeights[layer] = totalDerivWeights[layer].add(derivWeights[layer]);
        totalDerivBiases[layer] = totalDerivBiases[layer].add(derivBiases[layer]);
      }
    }

    for (let layer = 0; layer < this.weights.length; layer++) {
      // calculate the shift needed
      const layerGradientWeights = totalDerivWeights[layer].map(
        (x) => (x / batchSize) * this.learningRate,
      );
      const layerGradientBiases = totalDerivBiases[layer].map(
        (x) => (x / batchSize) * this.learningRate,
      );

      this.weights[layer] = this.weights[layer].subtract(layerGradientWeights);
      this.biases[layer] = this.biases[layer].subtract(layerGradientBiases);
    }

    return {
      averageCost: totalBatchCost / batchSize,
      accuracy: correct / batchSize,
    };
  }

  trainOne(
    inputs: number[],
    target: number[],
    label: U,
    dryRun?: false,
  ): {
    cost: number;
    isCorrect: boolean;
    derivWeights: Matrix[];
    derivBiases: Matrix[];
    derivCostWrtToInputs: number[];
  };
  trainOne(
    inputs: number[],
    target: number[],
    label: U,
    dryRun: true,
  ): {
    cost: number;
    isCorrect: boolean;
    derivCostWrtToInputs: number[];
  };

  trainOne(
    inputs: number[],
    target: number[],
    label: U,
    dryRun = false,
  ): {
    cost: number;
    isCorrect: boolean;
    derivWeights?: Matrix[];
    derivBiases?: Matrix[];
    derivCostWrtToInputs: number[];
  } {
    // create variables for caching
    const prenormalizedLayers: number[][] = [];
    const normalizedLayers: number[][] = [];

    // feed forward with caching
    let current: Matrix = Matrix.from1dArray(inputs);
    for (let layer = 0; layer < this.weights.length; layer++) {
      current = Matrix.multiply(this.weights[layer], current).add(this.biases[layer]);
      prenormalizedLayers.push(current.to1dArray()); // matrix (that is actually a vector) to number[]

      // only apply activation function if it isn't the last layer or there isn't an output layer activation function
      if (layer < this.weights.length - 1 || this.outputLayerActivationFn == null) {
        current = current.map(this.activationFn); // normalize by applying activation function
        normalizedLayers.push(current.to1dArray()); // vector to number[]
      }
    }

    const prenormalizedOut: number[] = current.to1dArray();
    const out =
      this.outputLayerActivationFn == null
        ? prenormalizedOut
        : this.outputLayerActivationFn(prenormalizedOut);

    // calculate cost and accuracy
    const cost = out.reduce((acc, actual, i) => acc + this.costFunction(target[i], actual), 0);
    const isCorrect = this.isCorrectFn(out) === label;

    // backpropagate
    let derivCostWrtToLayerNodes: number[] = [];
    if (this.outputLayerActivationFn == null) {
      derivCostWrtToLayerNodes = out.map((actual, i) => this.derivCostFunction(target[i], actual));
    } else if (
      this.outputLayerActivationFn === NeuralNetwork.outputLayerActivationFunctions.softmax[0] &&
      this.costFunction === NeuralNetwork.costFunctions.crossEntropy[0]
    ) {
      // this combination of cost function + output layer activation function makes the math easy (optimization improvement)
      derivCostWrtToLayerNodes = out.map((actual, i) => actual - target[i]);
    } else {
      const derivCostWrtOut = out.map((actual, i) => this.derivCostFunction(target[i], actual));

      const n = out.length;
      derivCostWrtToLayerNodes = Array(n).fill(0);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const derivOut_jWrtPrenormalizedOut_i = this.derivOutputLayerActivationFn!(
            out,
            prenormalizedOut,
            i,
            j,
          );
          derivCostWrtToLayerNodes[i] += derivCostWrtOut[j] * derivOut_jWrtPrenormalizedOut_i;
        }
      }
    }

    const derivWeights: Matrix[] = [];
    const derivBiases: Matrix[] = [];
    if (!dryRun) {
      for (let i = 1; i < this.structure.length; i++) {
        derivWeights.push(new Matrix(this.structure[i], this.structure[i - 1]));
        derivBiases.push(new Matrix(this.structure[i], 1));
      }
    }

    for (let layer = this.weights.length - 1; layer >= 0; layer--) {
      const derivCostWrtFromLayerNodes: number[] = Array(this.structure[layer]).fill(0);
      for (let to = 0; to < this.structure[layer + 1]; to++) {
        // partial derivative of cost with respect to bias
        const derivCostWrtBias_to =
          derivCostWrtToLayerNodes[to] *
          (layer < this.weights.length - 1 || this.outputLayerActivationFn == null
            ? this.derivActivationFn(prenormalizedLayers[layer][to])
            : 1);
        if (!dryRun) {
          derivBiases[layer].data[to][0] = derivCostWrtBias_to;
        }

        for (let from = 0; from < this.structure[layer]; from++) {
          if (!dryRun) {
            // partial derivative of cost with respect to weight
            const derivCostWrtWeight_to_from =
              derivCostWrtBias_to * (layer > 0 ? normalizedLayers[layer - 1][from] : inputs[from]);
            derivWeights[layer].data[to][from] += derivCostWrtWeight_to_from;
          }

          // part of the partial derivative of cost with respect to 'from layer node'
          const sumNode_toDerivCostWrtNode_from =
            derivCostWrtBias_to * this.weights[layer].data[to][from];
          derivCostWrtFromLayerNodes[from] += sumNode_toDerivCostWrtNode_from;
        }
      }

      derivCostWrtToLayerNodes = derivCostWrtFromLayerNodes;
    }

    return {
      cost,
      isCorrect,
      ...(!dryRun ? { derivWeights } : {}),
      ...(!dryRun ? { derivBiases } : {}),
      derivCostWrtToInputs: derivCostWrtToLayerNodes,
    };
  }

  testBatch(
    inputs: number[][],
    target: number[][],
    labels: U[],
  ): { averageCost: number; accuracy: number } {
    if (inputs.length !== target.length) {
      throw new Error(
        `Number of input samples [${inputs.length}] does not match number of target samples [${target.length}]`,
      );
    }
    const batchSize = inputs.length;

    let totalCost = 0;
    let correct = 0;
    for (let sample = 0; sample < batchSize; sample++) {
      const out: number[] = this.feedForward(inputs[sample]);
      if (this.isCorrectFn(out) === labels[sample]) correct++;
      totalCost += out.reduce(
        (acc, actual, i) => acc + this.costFunction(target[sample][i], actual),
        0,
      );
    }
    return {
      averageCost: totalCost / batchSize,
      accuracy: correct / batchSize,
    };
  }

  serialize(): string {
    const activationFn = this.serializeActivationFn();
    if (activationFn == null) {
      throw new Error(
        'Activation function could not be parsed. Please use an activation function from NeuralNetwork.activationFunctions.',
      );
    }

    const costFunction = this.serializeCostFunction();
    if (costFunction == null) {
      throw new Error(
        'Cost function could not be parsed. Please use a cost function from NeuralNetwork.costFunctions.',
      );
    }

    const outputLayerActivationFn = this.serializeOutputLayerActivationFn();
    if (outputLayerActivationFn == null) {
      throw new Error(
        'Output layer activation function could not be parsed. Please use an output layer activation function function from NeuralNetwork.outputLayerActivationFunctions.',
      );
    }

    const data: SerializedNetwork = {
      structure: this.structure,
      outputMappingFn: this.outputMappingFn.toString(),
      isCorrectFn: this.isCorrectFn.toString(),
      learningRate: this.learningRate,
      weights: this.weights.map((matrix) => matrix.toArray()),
      biases: this.biases.map((vector) => vector.toArray().map((row) => row[0])),
      activationFn,
      costFunction,
      outputLayerActivationFn,
    };
    return JSON.stringify(data);
  }

  protected serializeActivationFn(): string | null {
    for (const fnName in NeuralNetwork.activationFunctions) {
      const fnImpl = NeuralNetwork.activationFunctions[fnName][0];
      if (fnImpl === this.activationFn) return fnName;
    }
    return null;
  }

  protected serializeCostFunction(): string | null {
    for (const fnName in NeuralNetwork.costFunctions) {
      const fnImpl = NeuralNetwork.costFunctions[fnName][0];
      if (fnImpl === this.costFunction) return fnName;
    }
    return null;
  }

  protected serializeOutputLayerActivationFn(): string | null {
    for (const fnName in NeuralNetwork.outputLayerActivationFunctions) {
      const fnImpl = NeuralNetwork.outputLayerActivationFunctions[fnName][0];
      if (fnImpl === this.outputLayerActivationFn) return fnName;
    }
    return null;
  }

  static deserialize<T, U>(filename: string): NeuralNetwork<T, U> {
    const serializedData = fs.readFileSync(filename, { encoding: 'utf-8' });

    const [jsonParseErr, parsedData] = trySync(() => JSON.parse(serializedData));
    if (jsonParseErr || parsedData == undefined) throw jsonParseErr;

    const [zodParseErr, data] = trySync(() => serializedNetworkSchema.parse(parsedData));
    if (zodParseErr || data == undefined) throw zodParseErr;

    // deserialize outputMappingFn and isCorrectFn
    const [outputMappingFnParseErr, outputMappingFn] = trySync(() =>
      eval(`(${data.outputMappingFn})`),
    );
    if (outputMappingFnParseErr || outputMappingFn == undefined) throw outputMappingFnParseErr;

    const [isCorrectFnParseErr, isCorrectFn] = trySync(() => eval(`(${data.isCorrectFn})`));
    if (isCorrectFnParseErr || isCorrectFn == undefined) throw isCorrectFnParseErr;

    const nn = new NeuralNetwork<T, U>(
      data.structure,
      outputMappingFn as (nums: number[]) => T, // no way to verify
      isCorrectFn as (nums: number[]) => U, // no way to verify
      data.learningRate,
      NeuralNetwork.activationFunctions[data.activationFn],
      NeuralNetwork.costFunctions[data.costFunction],
      NeuralNetwork.outputLayerActivationFunctions[data.outputLayerActivationFn],
    );

    // copy weights/biases
    for (let layer = 0; layer < data.structure.length - 1; layer++) {
      nn.weights[layer] = Matrix.from2dArray(data.weights[layer]);
      nn.biases[layer] = Matrix.from1dArray(data.biases[layer]);
    }
    return nn;
  }
}

const activationFnSchema = z.enum(Object.keys(NeuralNetwork.activationFunctions));
const costFunctionSchema = z.enum(Object.keys(NeuralNetwork.costFunctions));
const outputLayerActivationFnSchema = z.enum(
  Object.keys(NeuralNetwork.outputLayerActivationFunctions),
);

const serializedNetworkSchema = z.object({
  structure: z.number().array(),
  outputMappingFn: z.string(),
  isCorrectFn: z.string(),
  learningRate: z.number(),
  weights: z.number().array().array().array(),
  biases: z.number().array().array(),
  activationFn: activationFnSchema,
  costFunction: costFunctionSchema,
  outputLayerActivationFn: outputLayerActivationFnSchema,
});

export type SerializedNetwork = z.infer<typeof serializedNetworkSchema>;
