/**
 * Calculates tokens per second over a sliding window (default 2 seconds).
 * Prevents the "starts low and slowly climbs" artifact caused by cumulative averages.
 */
export class SlidingWindowTpsCalculator {
  constructor(windowMs = 2000) {
    this.windowMs = windowMs;
    this.samples = []; // Array of { time: number, tokens: number }
  }

  reset() {
    this.samples = [];
  }

  /**
   * Adds a token sample.
   * @param {number} timestamp - Timestamp in ms
   * @param {number} cumulativeTokens - Accumulated token count
   */
  addSample(timestamp, cumulativeTokens) {
    this.samples.push({ time: timestamp, tokens: cumulativeTokens });
    this.prune(timestamp);
  }

  /**
   * Prunes samples older than windowMs, keeping at least one base sample prior to window cutoff if available.
   */
  prune(now) {
    const cutoff = now - this.windowMs;
    while (this.samples.length > 2 && this.samples[1].time <= cutoff) {
      this.samples.shift();
    }
  }

  /**
   * Computes the current token rate.
   * @param {number} now - Current timestamp in ms
   * @returns {number} TPS rate rounded to 1 decimal place, or 0 if insufficient samples/time.
   */
  getTps(now = Date.now()) {
    this.prune(now);
    if (this.samples.length < 2) {
      return 0;
    }

    const oldest = this.samples[0];
    const latest = this.samples[this.samples.length - 1];

    const timeDiffSec = (latest.time - oldest.time) / 1000;
    const tokenDiff = latest.tokens - oldest.tokens;

    if (timeDiffSec < 0.05 || tokenDiff < 0) {
      return 0;
    }

    const rawTps = tokenDiff / timeDiffSec;
    return Math.round(rawTps * 10) / 10;
  }
}
