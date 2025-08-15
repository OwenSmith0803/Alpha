// @ts-check

import eslint from '@eslint/js';
import spellcheck from 'eslint-plugin-spellcheck';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist', 'node_modules'] },
  {
    extends: [eslint.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.ts'],
    plugins: {
      '@typescript-eslint': tseslint.plugin,
      spellcheck,
    },
    languageOptions: {
      ecmaVersion: 15, // ES2024
      parser: tseslint.parser,
      parserOptions: {
        project: './tsconfig.json',
      },
    },
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          vars: 'all', // check all variables
          args: 'after-used', // check function arguments after used ones
          varsIgnorePattern: '^_', // ignore variables starting with _
          argsIgnorePattern: '^_', // ignore function args starting with _
        },
      ],
      'no-fallthrough': 'warn',
      'spellcheck/spell-checker': [
        'warn',
        {
          comments: true,
          strings: true,
          identifiers: true,
          templates: true,
          lang: 'en_US',
          skipWords: [
            'relu',
            'sigmoid',
            'softmax',
            'acc',
            'deriv',
            'wrt',
            'backpropagate',
            'impl',
            'zod',
            'utf',
            'enum',
            'nums',
            'mnist',
            'readline',
            'crlf',
            'dataset',
            'num',
            'iter',
            'datasets',
            'csv',
            'unadded'
          ],
          skipIfMatch: ['https?://[^\\s]*'],
          minLength: 3,
        },
      ],
    },
  },
);
