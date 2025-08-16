import fs from 'fs';
import z from 'zod';
import { Matrix } from './Matrix';
import {
  Network,
  NeuralNetwork,
  OutputLayerActivationFunction,
  OutputLayerActivationFunctionDeriv,
} from './NeuralNetwork';
import { MappingFn, trySync } from './utils/utils';

// T is the outputted type from the .predict() method (Image, e.g.). U is the label type (number, e.g.)
export class Decoder<T, U> extends NeuralNetwork<T, U> {
  private encoderModel: Network<U>;
  private encoderModelFilePath: string;
  constructor(
    structure: number[],
    outputMappingFn: (out: number[]) => T,
    [encoderModel, encoderModelFilePath]: [Network<U>, string],
    learningRate = 0.01,
    activationFunctions: [MappingFn, MappingFn] = NeuralNetwork.activationFunctions.relu,
    outputLayerActivationFunctions:
      | [OutputLayerActivationFunction, OutputLayerActivationFunctionDeriv]
      | [null, null] = [null, null],
  ) {
    if (
      structure[0] !== encoderModel.structure.at(-1) ||
      structure.at(-1) !== encoderModel.structure[0]
    ) {
      throw new Error(
        `The structure for a decoder model must be inverse to the structure of the inputted encoder model. Expected structure: ${encoderModel.structure.at(-1)}->${encoderModel.structure[0]}; got: ${structure[0]}->${structure.at(-1)}`,
      );
    }

    super(
      structure,
      outputMappingFn,
      (nums: number[]): U => {
        const generatedPixels = this.feedForward(nums);
        return encoderModel.predict(generatedPixels);
      },
      learningRate,
      activationFunctions,
      NeuralNetwork.costFunctions.meanSquaredError, // don't need
      outputLayerActivationFunctions,
    );

    this.encoderModel = encoderModel;
    this.encoderModelFilePath = encoderModelFilePath;
  }

  override trainOne(
    inputs: number[],
    target: number[],
    label: U,
    dryRun?: false,
  ): {
    cost: number;
    isCorrect: boolean;
    derivWeights: Matrix[];
    derivBiases: Matrix[];
    derivCostWrtInputs: number[];
  };
  override trainOne(
    inputs: number[],
    target: number[],
    label: U,
    dryRun: true,
  ): {
    cost: number;
    isCorrect: boolean;
    derivCostWrtInputs: number[];
  };

  override trainOne(
    inputs: number[],
    target: number[], // not needed
    label: U,
    dryRun = false,
  ): {
    cost: number;
    isCorrect: boolean;
    derivWeights?: Matrix[];
    derivBiases?: Matrix[];
    derivCostWrtInputs: number[];
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
    const { cost, isCorrect, derivCostWrtInputs } = this.encoderModel.trainOne(
      out,
      inputs,
      label,
      true,
    );

    // backpropagate
    let derivCostWrtToLayerNodes: number[] = [];
    if (this.outputLayerActivationFn == null) {
      derivCostWrtToLayerNodes = derivCostWrtInputs;
    } else {
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
          derivCostWrtToLayerNodes[i] += derivCostWrtInputs[j] * derivOut_jWrtPrenormalizedOut_i;
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
      derivCostWrtInputs: derivCostWrtToLayerNodes,
    };
  }

  override serialize(): string {
    const activationFn = this.serializeActivationFn();
    if (activationFn == null) {
      throw new Error(
        'Activation function could not be parsed. Please use an activation function from NeuralNetwork.activationFunctions.',
      );
    }

    const outputLayerActivationFn = this.serializeOutputLayerActivationFn();
    if (outputLayerActivationFn == null) {
      throw new Error(
        'Output layer activation function could not be parsed. Please use an output layer activation function function from NeuralNetwork.outputLayerActivationFunctions.',
      );
    }

    const data: SerializedDecoder = {
      structure: this.structure,
      outputMappingFn: this.outputMappingFn.toString(),
      encoderModelFilePath: this.encoderModelFilePath,
      learningRate: this.learningRate,
      weights: this.weights.map((matrix) => matrix.toArray()),
      biases: this.biases.map((vector) => vector.to1dArray()),
      activationFn,
      outputLayerActivationFn,
    };
    return JSON.stringify(data);
  }

  static override deserialize<T, U>(filename: string): Decoder<T, U> {
    const serializedData = fs.readFileSync(filename, { encoding: 'utf-8' });

    const [jsonParseErr, parsedData] = trySync(() => JSON.parse(serializedData));
    if (jsonParseErr || parsedData == undefined) throw jsonParseErr;

    const [zodParseErr, data] = trySync(() => serializedDecoderSchema.parse(parsedData));
    if (zodParseErr || data == undefined) throw zodParseErr;

    // deserialize outputMappingFn
    const [outputMappingFnParseErr, outputMappingFn] = trySync(() =>
      eval(`(${data.outputMappingFn})`),
    );
    if (outputMappingFnParseErr || outputMappingFn == undefined) throw outputMappingFnParseErr;

    // deserialize encoder model
    const [encoderModelDeserializeErr, encoderModel] = trySync(() =>
      NeuralNetwork.deserialize<U, U>(data.encoderModelFilePath),
    );
    if (encoderModelDeserializeErr || encoderModel == undefined) throw encoderModelDeserializeErr;

    const nn = new Decoder<T, U>(
      data.structure,
      outputMappingFn as (nums: number[]) => T,
      [encoderModel, data.encoderModelFilePath],
      data.learningRate,
      NeuralNetwork.activationFunctions[data.activationFn],
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
const outputLayerActivationFnSchema = z.enum(
  Object.keys(NeuralNetwork.outputLayerActivationFunctions),
);

const serializedDecoderSchema = z.object({
  structure: z.number().array(),
  outputMappingFn: z.string(),
  encoderModelFilePath: z.string(),
  learningRate: z.number(),
  weights: z.number().array().array().array(),
  biases: z.number().array().array(),
  activationFn: activationFnSchema,
  outputLayerActivationFn: outputLayerActivationFnSchema,
});

export type SerializedDecoder = z.infer<typeof serializedDecoderSchema>;
