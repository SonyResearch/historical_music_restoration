import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { historicalExamples } from "@/data/audioExamples";
import { mosPreservation, mosQuality } from "@/data/results";

describe("published research data", () => {
  it("provides the selected balanced historical examples", () => {
    expect(historicalExamples).toHaveLength(6);
    expect(historicalExamples.filter((item) => item.category === "Full-Orchestra")).toHaveLength(3);
    expect(historicalExamples.filter((item) => item.category === "Light Orchestra")).toHaveLength(3);
    expect(historicalExamples.every((item) => item.conditions.length === 6)).toBe(true);
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
  });
});
