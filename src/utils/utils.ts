export type MappingFn = (num: number) => number;

type Result<T> = [undefined, T] | [unknown];
export function trySync<T>(wrapperFn: () => T): Result<T> {
  try {
    return [undefined, wrapperFn()];
  } catch (e) {
    return [e];
  }
}

export async function tryAsync<T>(wrapperFn: () => Promise<T>): Promise<Result<T>> {
  try {
    return [undefined, await wrapperFn()];
  } catch (e) {
    return [e];
  }
}
