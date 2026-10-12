import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    rules: {
      // Leading underscore marks a deliberately unused parameter.
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
      // Vite's own template default: a module exporting a helper beside its
      // component only loses hot-reload state, so this is advisory.
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
      // React Compiler readiness rules added in eslint-plugin-react-hooks v7.
      // They flag pre-existing fetch-on-mount and reset-on-prop-change effects
      // that work today; rewriting them is UI work to be done and tested
      // deliberately, not as a lint fix. Kept visible as warnings.
      'react-hooks/set-state-in-effect': 'warn',
      'react-hooks/purity': 'warn',
      'react-hooks/immutability': 'warn',
    },
  },
])
