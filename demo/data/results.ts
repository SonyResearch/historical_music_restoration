import type { MosResult } from "@/types/results";

export const mosQuality: MosResult[] = [
  { method: "INPUT", label: "Input", mean: 1.170, ciLow: 1.055, ciHigh: 1.286 },
  { method: "BABE2_PRETRAINED", label: "BABE2-P", mean: 3.523, ciLow: 3.243, ciHigh: 3.803 },
  { method: "BABE2_FOS", label: "BABE2-FOS", mean: 2.716, ciLow: 2.473, ciHigh: 2.959 },
  { method: "CFM40_GRAMOPHONE_ONLY", label: "SAMECFM-G", mean: 3.216, ciLow: 2.966, ciHigh: 3.466 },
  { method: "CFM40", label: "SAMECFM", mean: 3.886, ciLow: 3.706, ciHigh: 4.066 },
  { method: "GROUND_TRUTH", label: "Ground truth", mean: 4.443, ciLow: 4.298, ciHigh: 4.588 },
];

export const mosPreservation: MosResult[] = [
  { method: "BABE2_PRETRAINED", label: "BABE2-P", mean: 326 / 88, ciLow: 3.340, ciHigh: 4.069 },
  { method: "BABE2_FOS", label: "BABE2-FOS", mean: 3.898, ciLow: 3.616, ciHigh: 4.179 },
  { method: "CFM40_GRAMOPHONE_ONLY", label: "SAMECFM-G", mean: 4.261, ciLow: 3.995, ciHigh: 4.527 },
  { method: "CFM40", label: "SAMECFM", mean: 4.318, ciLow: 4.057, ciHigh: 4.579 },
];
