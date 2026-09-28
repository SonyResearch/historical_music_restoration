import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { historicalExamples, rewardFineTuningExamples } from "@/data/audioExamples";
import { mosPreservation, mosQuality } from "@/data/results";

describe("published research data", () => {
  it("provides the selected balanced historical examples", () => {
    expect(historicalExamples).toHaveLength(6);
    expect(historicalExamples.filter((item) => item.category === "Full-Orchestra")).toHaveLength(3);
    expect(historicalExamples.filter((item) => item.category === "Light Orchestra")).toHaveLength(3);
    expect(historicalExamples.every((item) => item.conditions.length === 6)).toBe(true);
  });

  it("provides every MOS-Q sample for compound-reward comparison", () => {
    expect(rewardFineTuningExamples).toHaveLength(10);
    expect(rewardFineTuningExamples.filter((item) => item.category === "Full-Orchestra")).toHaveLength(5);
    expect(rewardFineTuningExamples.filter((item) => item.category === "Light Orchestra")).toHaveLength(5);
    expect(rewardFineTuningExamples.every((item) => item.conditions.length === 3)).toBe(true);
    expect(rewardFineTuningExamples.every((item) => item.conditions.some((condition) => condition.id === "aa-pq-songbench"))).toBe(true);
  });

  it("matches the validated subjective sensitivity analysis", () => {
    expect(mosQuality.find((row) => row.method === "CFM40")?.mean).toBe(3.886);
    expect(mosPreservation.find((row) => row.method === "CFM40")?.mean).toBe(4.318);
  });

  it("ships every referenced media file", () => {
    for (const example of historicalExamples) {
      for (const condition of example.conditions) {
        expect(fs.existsSync(path.join(process.cwd(), "public", condition.src))).toBe(true);
      }
    }
    for (const example of rewardFineTuningExamples) {
      for (const condition of example.conditions) {
        expect(fs.existsSync(path.join(process.cwd(), "public", condition.src))).toBe(true);
      }
    }
  });
});
