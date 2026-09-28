import { AudioDemoGrid } from "@/components/audio/AudioDemoGrid";
import { PlaybackProvider } from "@/components/audio/PlaybackProvider";
import { DatasetSummary } from "@/components/dataset/DatasetSummary";
import { Footer } from "@/components/Footer";
import { Hero } from "@/components/Hero";
import { DegradationDiagram } from "@/components/method/DegradationDiagram";
import { PipelineDiagram } from "@/components/method/PipelineDiagram";
import { Navbar } from "@/components/Navbar";
import { PaperSection } from "@/components/PaperSection";
import { MOSResults } from "@/components/results/MOSResults";
import { historicalExamples } from "@/data/audioExamples";

export default function Home() {
  return (
    <PlaybackProvider>
      <Navbar />
      <main>
        <Hero />

        <section className="section audio-section" id="demo" aria-labelledby="demo-title">
          <div className="container">
            <div className="section-heading"><p className="eyebrow">Audio demonstration</p><h2 id="demo-title">Listen to the restoration</h2><p>Switch between versions while playback remains synchronized. Headphones recommended.</p></div>
            <AudioDemoGrid examples={historicalExamples} />
          </div>
        </section>

        <section className="section motivation-section" aria-labelledby="motivation-title">
          <div className="container">
            <div className="section-heading"><p className="eyebrow">The problem</p><h2 id="motivation-title">Why historical music restoration?</h2></div>
            <div className="motivation-copy">
              <p className="lead">Many of the greatest musicians&apos; works are hard to fully appreciate because they survive only through poor acoustic recordings.</p>
              <p>Beginning with the phonograph in 1877, performances were captured mechanically onto cylinders and later 78-rpm discs. Electrical recording became widespread around 1925 and rapidly improved fidelity, but earlier recordings retained interacting bandwidth loss, coloration, distortion, hiss, and clicks. Existing methods usually address only a subset of these historical degradations.</p>
            </div>
            <DegradationDiagram />
          </div>
        </section>

        <section className="section method-section" id="method" aria-labelledby="method-title">
          <div className="container">
            <div className="section-heading"><p className="eyebrow">Method</p><h2 id="method-title">Synthetic degradation + latent restoration</h2><p>SAMECFM is trained with Full-Orchestra + Section (FOS) data. It learns a conditional velocity field between degraded and clean SAME-L latents, then decodes the restored representation at 44.1 kHz.</p></div>
            <PipelineDiagram />
          </div>
        </section>

        <section className="section results-section" id="results" aria-labelledby="results-title">
          <div className="container">
            <div className="section-heading"><p className="eyebrow">Evaluation</p><h2 id="results-title">Subjective listening results</h2><p>MOS-Q measures overall audio quality; MOS-P measures preservation of the essential musical content.</p></div>
            <MOSResults />
          </div>
        </section>

        <DatasetSummary />
        <PaperSection />
      </main>
      <Footer />
    </PlaybackProvider>
  );
}
