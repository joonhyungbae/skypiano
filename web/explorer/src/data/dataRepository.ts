import { getDateAdapter } from '@/data/adapters';
import { withBase } from '@/utils/assetUrl';
import type {
  DatasetManifest,
  DatasetManifestEntry,
  NormalizedDataItem,
  RawDataItem,
  SampleTrialEntry
} from '@/data/types';

export interface DatasetManifestV2 extends DatasetManifest {
  samples?: SampleTrialEntry[];
}

const DEFAULT_MANIFEST: DatasetManifest = {
  version: 1,
  generatedAt: '',
  datasets: [
    {
      date: '220819',
      label: '2022.08.19',
      status: 'ready',
      basePath: '/Data/Processed/220819',
      dataFile: '/Data/final_data_220819.json',
      adapter: '220819',
      modalities: { video: true, audio: true, motion: true, midi: true, score: true }
    }
  ]
};

const cacheBust = () => `t=${Date.now()}`;

const inferDateFromItem = (item: RawDataItem): string => {
  const fromRecording = item.recordingName?.slice(0, 6);
  if (/^\d{6}$/.test(fromRecording)) {
    return fromRecording;
  }
  return 'unknown';
};

const normalizeItems = (
  rows: RawDataItem[],
  dataset: DatasetManifestEntry
): NormalizedDataItem[] => {
  const adapter = getDateAdapter(dataset.adapter);
  return rows.map((item) => {
    const dateKey = inferDateFromItem(item) !== 'unknown' ? inferDateFromItem(item) : dataset.date;
    return adapter.normalize(item, dateKey, dataset.basePath);
  });
};

/** Rewrites the manifest's site-absolute paths for the deployment base URL. */
const rebaseManifest = (manifest: DatasetManifestV2): DatasetManifestV2 => ({
  ...manifest,
  datasets: manifest.datasets.map((entry) => ({
    ...entry,
    basePath: withBase(entry.basePath),
    dataFile: entry.dataFile ? withBase(entry.dataFile) : entry.dataFile
  })),
  samples: manifest.samples?.map((entry) => ({
    ...entry,
    basePath: withBase(entry.basePath)
  }))
});

export const loadManifest = async (): Promise<DatasetManifestV2> => {
  try {
    const response = await fetch(withBase(`/Data/manifest.json?${cacheBust()}`));
    if (!response.ok) {
      return rebaseManifest(DEFAULT_MANIFEST);
    }
    return rebaseManifest((await response.json()) as DatasetManifestV2);
  } catch (error) {
    console.warn('Failed to load manifest. Using defaults.', error);
    return rebaseManifest(DEFAULT_MANIFEST);
  }
};

export const loadSampleTrials = async (): Promise<SampleTrialEntry[]> => {
  const manifest = await loadManifest();
  return manifest.samples ?? [];
};

const loadCombinedFallback = async (): Promise<RawDataItem[]> => {
  const response = await fetch(withBase(`/Data/final_data.json?${cacheBust()}`));
  if (!response.ok) {
    throw new Error('Failed to load /Data/final_data.json');
  }
  return (await response.json()) as RawDataItem[];
};

export const loadDateData = async (
  manifest: DatasetManifest,
  date: string
): Promise<NormalizedDataItem[]> => {
  const dataset = manifest.datasets.find((entry) => entry.date === date);
  if (!dataset || dataset.status !== 'ready') {
    return [];
  }

  try {
    const response = await fetch(`${dataset.dataFile}?${cacheBust()}`);
    if (!response.ok) {
      throw new Error(`Failed to load ${dataset.dataFile}`);
    }

    const rows = (await response.json()) as RawDataItem[];
    return normalizeItems(rows, dataset);
  } catch (error) {
    // Backward compatible fallback: combined file + date filtering
    const combined = await loadCombinedFallback();
    const filtered = combined.filter((item) => inferDateFromItem(item) === date);
    return normalizeItems(filtered, dataset);
  }
};

export const loadAllReadyDateData = async (
  manifest: DatasetManifest
): Promise<NormalizedDataItem[]> => {
  const readySets = manifest.datasets.filter((entry) => entry.status === 'ready');
  const results = await Promise.all(
    readySets.map(async (entry) => {
      return loadDateData(manifest, entry.date);
    })
  );

  return results.flat();
};
