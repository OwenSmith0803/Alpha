import { MappingFn } from './utils/utils';

export class Matrix {
  data: number[][] = [];
  constructor(
    private rows: number,
    private cols: number,
    mappingFn = (_i: number, _j: number) => 0,
  ) {
    this.data = Array.from({ length: rows }, (_, i) =>
      Array.from({ length: cols }, (_, j) => mappingFn(i, j)),
    );
  }

  static initRandom(_i: number, _j: number): number {
    return Math.random() * 2 - 1;
  }

  static from1dArray(nums: number[]): Matrix {
    const n = nums.length;
    return new Matrix(n, 1, (i: number, _: number) => nums[i]); // a vector of length n
  }

  static from2dArray(nums: number[][]): Matrix {
    const m = nums.length;
    const n = nums[0].length;
    return new Matrix(m, n, (i: number, j: number) => nums[i][j]);
  }

  toArray(): number[][] {
    return this.data.map((row) => row.slice()); // copy of data
  }

  to1dArray(): number[] {
    if (this.cols !== 1) {
      throw new Error('Matrix must have only 1 column to use .to1dArray()');
    }

    return this.data.map((row) => row[0]);
  }

  static multiply(m1: Matrix, m2: Matrix): Matrix {
    return m1.multiply(m2);
  }

  multiply(other: Matrix): Matrix {
    if (this.cols !== other.rows) {
      throw new Error(
        `Attempted to multiply ${this.rows}x${this.cols} matrix and ${other.rows}x${other.cols}: Cols of matrix 1 must match rows of matrix 2`,
      );
    }
    const multiplyLength = this.cols;

    return new Matrix(this.rows, other.cols, (i: number, j: number) => {
      let sum = 0;
      for (let k = 0; k < multiplyLength; k++) {
        sum += this.data[i][k] * other.data[k][j];
      }
      return sum;
    });
  }

  static add(m1: Matrix, m2: Matrix): Matrix {
    return m1.add(m2);
  }

  add(other: Matrix): Matrix {
    if (this.rows !== other.rows || this.cols !== other.cols) {
      throw new Error(
        `Matrix of size ${this.rows}x${this.cols} must match matrix of size ${other.rows}x${other.cols}`,
      );
    }

    return new Matrix(
      this.rows,
      this.cols,
      (i: number, j: number) => this.data[i][j] + other.data[i][j],
    );
  }

  static subtract(m1: Matrix, m2: Matrix): Matrix {
    return m1.subtract(m2);
  }

  subtract(other: Matrix): Matrix {
    if (this.rows !== other.rows || this.cols !== other.cols) {
      throw new Error(
        `Matrix of size ${this.rows}x${this.cols} must match matrix of size ${other.rows}x${other.cols}`,
      );
    }

    return new Matrix(
      this.rows,
      this.cols,
      (i: number, j: number) => this.data[i][j] - other.data[i][j],
    );
  }

  map(mappingFn: MappingFn): Matrix {
    return new Matrix(this.rows, this.cols, (i: number, j: number) => mappingFn(this.data[i][j]));
  }

  reduce(fn: (acc: number, val: number, i: number, j: number) => number, initVal = 0): number {
    let acc = initVal;
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        acc = fn(acc, this.data[i][j], i, j);
      }
    }
    return acc;
  }
}
