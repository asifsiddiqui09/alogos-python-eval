import { readFileSync } from 'fs';
import { SyntheticImageDetector } from './src/detector';
import { ImageData } from './src/types';

const binPath = process.argv[2];
const metaPath = binPath.replace(/\.bin$/, '.json');

const meta = JSON.parse(readFileSync(metaPath, 'utf-8'));
const raw = readFileSync(binPath);
const data = new Uint8ClampedArray(raw);

const imageData: ImageData = { width: meta.width, height: meta.height, data };

const detector = new SyntheticImageDetector();
const r = detector.analyse(imageData);

console.log(JSON.stringify({
  raw_score: r.rawScore,
  confidence: r.confidence,
  is_synthetic: r.isSynthetic,
  primary_variance: r.metadata.primaryVariance,
  coherence: r.metadata.coherence,
  pixels: r.metadata.pixelsAnalysed,
}));
