export {};

declare global {
  interface Array<T> {
    chunk: (chunkSize: number, includeLast: boolean) => T[][];
    shuffle: () => void;
  }
}

Array.prototype.chunk = function chunk<T>(this: T[], chunkSize: number, includeLast = true): T[][] {
  const result: T[][] = [];
  for (let i = 0; i <= this.length; i += chunkSize) {
    result.push(this.slice(i, i + chunkSize));
  }

  if (!includeLast && result.at(-1)?.length !== chunkSize) {
    result.pop();
  }
  return result;
};
