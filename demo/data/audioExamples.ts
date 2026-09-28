import type { AudioCondition, AudioExample } from "@/types/audio";

const historicalConditions = (id: string): AudioCondition[] => [
  { id: "input", label: "Historical Input", shortLabel: "Input", src: `/audio/historical/${id}/input.mp3` },
  { id: "behm-p", label: "BEHM-GAN pretrained", shortLabel: "BEHM-P", src: `/audio/historical/${id}/behm-p.mp3` },
  { id: "behm-fos", label: "BEHM-GAN FOS", shortLabel: "BEHM-FOS", src: `/audio/historical/${id}/behm-fos.mp3` },
  { id: "babe2-p", label: "BABE2 pretrained", shortLabel: "BABE2-P", src: `/audio/historical/${id}/babe2-p.mp3` },
  { id: "babe2-fos", label: "BABE2 FOS", shortLabel: "BABE2-FOS", src: `/audio/historical/${id}/babe2-fos.mp3` },
  { id: "samecfm", label: "SAMECFM (Ours)", shortLabel: "SAMECFM", src: `/audio/historical/${id}/samecfm.mp3`, isOurs: true },
];

export const historicalExamples: AudioExample[] = [
  { id: "ballet-egyptian", title: "Ballet Egyptian, Nos. 1 and 2", artist: "American Symphony Orchestra (Edison ensemble)", artistYears: "documented c. 1910s–1920s", recordLabel: "Edison", year: 1915, category: "Full-Orchestra", detail: "110–115 s", sourceUrl: "https://archive.org/details/78_ballet-egyptian-nos-1-and-2_american-symphony-orchestra-alexandre-luigini_gbia0078132a", conditions: historicalConditions("ballet-egyptian") },
  { id: "rigoletto", title: "Rigoletto Selection, Part 1", artist: "American Symphony Orchestra (Edison ensemble)", artistYears: "documented c. 1910s–1920s", recordLabel: "Edison", year: 1917, category: "Full-Orchestra", detail: "75–80 s", sourceUrl: "https://archive.org/details/78_rigoletto-selection-part-1_american-symphony-orchestra-g-verdi_gbia0298845a", conditions: historicalConditions("rigoletto") },
  { id: "hungarian-dance", title: "Hungarian Dance No. 5", artist: "Philadelphia Orchestra · Leopold Stokowski", artistYears: "1900–present", recordLabel: "Victor", year: 1917, category: "Full-Orchestra", detail: "95–100 s", sourceUrl: "https://archive.org/details/78_hungarian-dance-no-5_philadelphia-symphony-orchestra-brahms-leopold-stokowski_gbia0183029a", conditions: historicalConditions("hungarian-dance") },
  { id: "minuetto", title: "Minuetto", artist: "American Symphony Orchestra (Edison ensemble)", artistYears: "documented c. 1910s–1920s", recordLabel: "Edison", year: 1917, category: "Light Orchestra", detail: "80–85 s", sourceUrl: "https://archive.org/details/78_minuetto_american-symphony-orchestra-g-bolzoni_gbia0078138a", conditions: historicalConditions("minuetto") },
  { id: "scented-violets", title: "Scented Violets", artist: "Peerless Orchestra", artistYears: "documented 1913–1929", recordLabel: "Edison", year: 1920, category: "Light Orchestra", detail: "50–55 s", sourceUrl: "https://archive.org/details/78_scented-violets_peerless-orchestra-jules-reynard_gbia0178673b", conditions: historicalConditions("scented-violets") },
  { id: "lady-bird", title: "Lady Bird Tango", artist: "Peerless Orchestra", artistYears: "documented 1913–1929", recordLabel: "Zonophone", year: 1914, category: "Light Orchestra", detail: "95–100 s", sourceUrl: "https://archive.org/details/78_lady-bird-tango_the-peerless-orchestra-p-s-robinson_gbia3038426b", conditions: historicalConditions("lady-bird") },
];
