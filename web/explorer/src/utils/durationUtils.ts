// Shared duration utilities — consolidates logic that was duplicated across status/home cards

export interface HasDuration {
  duration: string;
}

// "MM:SS" values summed in minutes, returned as an "Xh Ym" or "Ym" string
export function calculateTotalDuration(items: HasDuration[]): string {
  let totalMinutes = 0;
  for (const item of items) {
    if (!item.duration) continue;
    const parts = item.duration.split(':');
    if (parts.length !== 2) continue;
    const minutes = parseInt(parts[0], 10) || 0;
    const seconds = parseInt(parts[1], 10) || 0;
    totalMinutes += minutes + Math.round(seconds / 60);
  }
  const hours = Math.floor(totalMinutes / 60);
  const remaining = totalMinutes % 60;
  return hours > 0 ? `${hours}h ${remaining}m` : `${remaining}m`;
}

// "MM:SS" → "Xm Ys" / "Xm" / "Ys" for display
export function formatDuration(duration: string): string {
  if (!duration) return '';
  const parts = duration.split(':');
  if (parts.length === 2) {
    const minutes = parseInt(parts[0], 10);
    const seconds = parseInt(parts[1], 10);
    if (minutes > 0 && seconds > 0) return `${minutes}m ${seconds}s`;
    if (minutes > 0) return `${minutes}m`;
    if (seconds > 0) return `${seconds}s`;
  }
  return duration;
}
