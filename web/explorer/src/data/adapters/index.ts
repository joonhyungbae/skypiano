import { adapter220819 } from '@/data/adapters/adapter220819';
import type { DateAdapter } from '@/data/types';

const passthroughAdapter: DateAdapter = {
  id: 'default',
  normalize(item, dateKey) {
    return {
      ...item,
      dateKey,
      assets: {}
    };
  }
};

const adapterMap: Record<string, DateAdapter> = {
  [adapter220819.id]: adapter220819
};

export const getDateAdapter = (adapterId: string): DateAdapter => {
  return adapterMap[adapterId] ?? passthroughAdapter;
};
