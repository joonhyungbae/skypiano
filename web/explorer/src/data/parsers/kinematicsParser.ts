import type { KinematicsData, KinematicsSample } from '@/data/types';

/**
 * Visual3D CSV: 1st column is time (seconds); the rest are channel values (joint angles, etc.).
 * Assumes a default sampling rate of 120Hz — fps is estimated back from the time axis.
 */
export async function loadKinematics({
  path,
  trialId,
}: {
  path?: string;
  trialId: string;
}): Promise<KinematicsData> {
  if (!path) {
    return { trialId, channelNames: [], samples: [], fps: 0 };
  }

  let text: string;
  try {
    const res = await fetch(path);
    if (!res.ok) return { trialId, channelNames: [], samples: [], fps: 0 };
    text = await res.text();
  } catch {
    return { trialId, channelNames: [], samples: [], fps: 0 };
  }

  const rows = text.split(/\r?\n/).filter((l) => l.length > 0);
  if (rows.length < 2) return { trialId, channelNames: [], samples: [], fps: 0 };

  const header = rows[0].split(',').map((h) => h.trim());
  const timeIdx = header.findIndex((h) => /^(time|time_sec|t)$/i.test(h));
  const channelCols = header
    .map((name, i) => ({ name, i }))
    .filter((c) => c.i !== timeIdx);

  const channelNames = channelCols.map((c) => c.name);
  const samples: KinematicsSample[] = [];

  for (let r = 1; r < rows.length; r++) {
    const cells = rows[r].split(',');
    const time = parseFloat(cells[timeIdx]);
    if (!Number.isFinite(time)) continue;

    const channels: Record<string, number> = {};
    for (const c of channelCols) {
      const v = parseFloat(cells[c.i]);
      if (Number.isFinite(v)) channels[c.name] = v;
    }
    samples.push({ time, channels });
  }

  const fps =
    samples.length >= 2
      ? Math.round(1 / (samples[1].time - samples[0].time))
      : 0;

  return { trialId, channelNames, samples, fps };
}
