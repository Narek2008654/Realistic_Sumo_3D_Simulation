/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: {
          0: 'var(--bg-0)',
          1: 'var(--bg-1)',
          2: 'var(--bg-2)',
          3: 'var(--bg-3)',
        },
        line: { DEFAULT: 'var(--line)', 2: 'var(--line-2)' },
        fg: { 0: 'var(--fg-0)', 1: 'var(--fg-1)', 2: 'var(--fg-2)' },
        accent: {
          DEFAULT: 'var(--accent)',
          hot: 'var(--accent-hot)',
          dim: 'var(--accent-dim)',
        },
        cyan: { DEFAULT: 'var(--cyan)', dim: 'var(--cyan-dim)' },
        win: 'var(--win)',
        loss: 'var(--loss)',
        warn: 'var(--warn)',
        idle: 'var(--idle)',
      },
      fontFamily: {
        display: ['Oxanium', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
      borderRadius: { DEFAULT: 'var(--radius)' },
      boxShadow: {
        panel: 'var(--shadow)',
        glow: '0 0 18px var(--accent-glow)',
        'glow-cyan': '0 0 16px var(--cyan-glow)',
      },
      letterSpacing: { label: '.12em', wide: '.06em' },
    },
  },
  plugins: [],
};
