import { describe, it, expect, beforeEach } from 'vitest';
import { SlidingWindowTpsCalculator } from './tpsCalculator';

describe('SlidingWindowTpsCalculator', () => {
  let calc;

  beforeEach(() => {
    calc = new SlidingWindowTpsCalculator(2000);
  });

  it('returns 0 when there are fewer than 2 samples', () => {
    expect(calc.getTps(1000)).toBe(0);
    calc.addSample(1000, 1);
    expect(calc.getTps(1000)).toBe(0);
  });

  it('calculates a constant rate for a constant token cadence from 2nd token onward', () => {
    // 1 token every 100ms => 10 tokens/sec
    // 1st token at 500ms
    calc.addSample(500, 1);
    expect(calc.getTps(500)).toBe(0);

    // 2nd token at 600ms (100ms elapsed) -> 1 tok / 0.1s = 10 tok/s
    calc.addSample(600, 2);
    expect(calc.getTps(600)).toBe(10);

    // 3rd token at 700ms (200ms elapsed) -> 2 tok / 0.2s = 10 tok/s
    calc.addSample(700, 3);
    expect(calc.getTps(700)).toBe(10);

    // 10th token at 1400ms (900ms elapsed) -> 9 tok / 0.9s = 10 tok/s
    calc.addSample(1400, 10);
    expect(calc.getTps(1400)).toBe(10);
  });

  it('maintains rate accuracy over a long stream via sliding window pruning', () => {
    // 1 token every 100ms => 10 tokens/sec
    for (let i = 1; i <= 50; i++) {
      const time = 500 + i * 100;
      calc.addSample(time, i);
      if (i >= 2) {
        expect(calc.getTps(time)).toBe(10);
      }
    }
  });

  it('resets samples properly', () => {
    calc.addSample(500, 1);
    calc.addSample(600, 2);
    expect(calc.getTps(600)).toBe(10);
    calc.reset();
    expect(calc.getTps(600)).toBe(0);
  });
});
