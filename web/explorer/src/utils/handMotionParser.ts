// Hand joint types
export interface HandJoint {
  x: number;
  y: number;
  z: number;
}

// Hand data structure
export interface HandFrame {
  leftHand: {
    end: HandJoint;
    hmid: HandJoint;
    id: HandJoint;
    if: HandJoint;
    im: HandJoint;
    ip: HandJoint;
    elbow: HandJoint;
    law: HandJoint;
    ld: HandJoint;
    lf: HandJoint;
    lm: HandJoint;
    lp: HandJoint;
    md: HandJoint;
    mee: HandJoint;
    wrist: HandJoint;
    mf: HandJoint;
    mm: HandJoint;
    mp: HandJoint;
    rd: HandJoint;
    rf: HandJoint;
    rm: HandJoint;
    rp: HandJoint;
    sh: HandJoint;
    td: HandJoint;
    tf: HandJoint;
    tm: HandJoint;
  };
  rightHand: {
    end: HandJoint;
    hmid: HandJoint;
    id: HandJoint;
    if: HandJoint;
    im: HandJoint;
    ip: HandJoint;
    elbow: HandJoint;
    law: HandJoint;
    ld: HandJoint;
    lf: HandJoint;
    lm: HandJoint;
    lp: HandJoint;
    md: HandJoint;
    wrist: HandJoint;
    mf: HandJoint;
    mm: HandJoint;
    mp: HandJoint;
    rd: HandJoint;
    rf: HandJoint;
    rm: HandJoint;
    rp: HandJoint;
    sh: HandJoint;
    td: HandJoint;
    tf: HandJoint;
    tm: HandJoint;
  };
  frameIndex: number;
}

// Parse CSV data
export function parseHandMotionCSV(csvText: string): HandFrame[] {
  const lines = csvText.trim().split('\n');
  const headers = lines[0].split(',');
  const frames: HandFrame[] = [];

  // Build the index mapping from the header
  const headerMap = new Map<string, number>();
  headers.forEach((header, index) => {
    headerMap.set(header.trim(), index);
  });

  // Parse each data row
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(val => {
      const trimmed = val.trim();
      if (trimmed === '' || trimmed.toLowerCase() === 'nan') {
        return NaN;
      }
      const parsed = parseFloat(trimmed);
      return isNaN(parsed) ? NaN : parsed;
    });
    
    const frame: HandFrame = {
      leftHand: {
        end: {
          x: values[headerMap.get('LEnd_x') || 0],
          y: values[headerMap.get('LEnd_y') || 0],
          z: values[headerMap.get('LEnd_z') || 0]
        },
        hmid: {
          x: values[headerMap.get('LHMID_x') || 0],
          y: values[headerMap.get('LHMID_y') || 0],
          z: values[headerMap.get('LHMID_z') || 0]
        },
        id: {
          x: values[headerMap.get('LID_x') || 0],
          y: values[headerMap.get('LID_y') || 0],
          z: values[headerMap.get('LID_z') || 0]
        },
        if: {
          x: values[headerMap.get('LIF_x') || 0],
          y: values[headerMap.get('LIF_y') || 0],
          z: values[headerMap.get('LIF_z') || 0]
        },
        im: {
          x: values[headerMap.get('LIM_x') || 0],
          y: values[headerMap.get('LIM_y') || 0],
          z: values[headerMap.get('LIM_z') || 0]
        },
        ip: {
          x: values[headerMap.get('LIP_x') || 0],
          y: values[headerMap.get('LIP_y') || 0],
          z: values[headerMap.get('LIP_z') || 0]
        },
        elbow: {
          x: values[headerMap.get('LElbow_x') || 0],
          y: values[headerMap.get('LElbow_y') || 0],
          z: values[headerMap.get('LElbow_z') || 0]
        },
        law: {
          x: values[headerMap.get('LLAW_x') || 0],
          y: values[headerMap.get('LLAW_y') || 0],
          z: values[headerMap.get('LLAW_z') || 0]
        },
        ld: {
          x: values[headerMap.get('LLD_x') || 0],
          y: values[headerMap.get('LLD_y') || 0],
          z: values[headerMap.get('LLD_z') || 0]
        },
        lf: {
          x: values[headerMap.get('LLF_x') || 0],
          y: values[headerMap.get('LLF_y') || 0],
          z: values[headerMap.get('LLF_z') || 0]
        },
        lm: {
          x: values[headerMap.get('LLM_x') || 0],
          y: values[headerMap.get('LLM_y') || 0],
          z: values[headerMap.get('LLM_z') || 0]
        },
        lp: {
          x: values[headerMap.get('LLP_x') || 0],
          y: values[headerMap.get('LLP_y') || 0],
          z: values[headerMap.get('LLP_z') || 0]
        },
        md: {
          x: values[headerMap.get('LMD_x') || 0],
          y: values[headerMap.get('LMD_y') || 0],
          z: values[headerMap.get('LMD_z') || 0]
        },
        mee: {
          x: values[headerMap.get('LMEE_x') || 0],
          y: values[headerMap.get('LMEE_y') || 0],
          z: values[headerMap.get('LMEE_z') || 0]
        },
        wrist: {
          x: values[headerMap.get('LWrist_x') || 0],
          y: values[headerMap.get('LWrist_y') || 0],
          z: values[headerMap.get('LWrist_z') || 0]
        },
        mf: {
          x: values[headerMap.get('LMF_x') || 0],
          y: values[headerMap.get('LMF_y') || 0],
          z: values[headerMap.get('LMF_z') || 0]
        },
        mm: {
          x: values[headerMap.get('LMM_x') || 0],
          y: values[headerMap.get('LMM_y') || 0],
          z: values[headerMap.get('LMM_z') || 0]
        },
        mp: {
          x: values[headerMap.get('LMP_x') || 0],
          y: values[headerMap.get('LMP_y') || 0],
          z: values[headerMap.get('LMP_z') || 0]
        },
        rd: {
          x: values[headerMap.get('LRD_x') || 0],
          y: values[headerMap.get('LRD_y') || 0],
          z: values[headerMap.get('LRD_z') || 0]
        },
        rf: {
          x: values[headerMap.get('LRF_x') || 0],
          y: values[headerMap.get('LRF_y') || 0],
          z: values[headerMap.get('LRF_z') || 0]
        },
        rm: {
          x: values[headerMap.get('LRM_x') || 0],
          y: values[headerMap.get('LRM_y') || 0],
          z: values[headerMap.get('LRM_z') || 0]
        },
        rp: {
          x: values[headerMap.get('LRP_x') || 0],
          y: values[headerMap.get('LRP_y') || 0],
          z: values[headerMap.get('LRP_z') || 0]
        },
        sh: {
          x: values[headerMap.get('LSh_x') || 0],
          y: values[headerMap.get('LSh_y') || 0],
          z: values[headerMap.get('LSh_z') || 0]
        },
        td: {
          x: values[headerMap.get('LTD_x') || 0],
          y: values[headerMap.get('LTD_y') || 0],
          z: values[headerMap.get('LTD_z') || 0]
        },
        tf: {
          x: values[headerMap.get('LTF_x') || 0],
          y: values[headerMap.get('LTF_y') || 0],
          z: values[headerMap.get('LTF_z') || 0]
        },
        tm: {
          x: values[headerMap.get('LTM_x') || 0],
          y: values[headerMap.get('LTM_y') || 0],
          z: values[headerMap.get('LTM_z') || 0]
        }
      },
      rightHand: {
        end: {
          x: values[headerMap.get('REnd_x') || 0],
          y: values[headerMap.get('REnd_y') || 0],
          z: values[headerMap.get('REnd_z') || 0]
        },
        hmid: {
          x: values[headerMap.get('RHMID_x') || 0],
          y: values[headerMap.get('RHMID_y') || 0],
          z: values[headerMap.get('RHMID_z') || 0]
        },
        id: {
          x: values[headerMap.get('RID_x') || 0],
          y: values[headerMap.get('RID_y') || 0],
          z: values[headerMap.get('RID_z') || 0]
        },
        if: {
          x: values[headerMap.get('RIF_x') || 0],
          y: values[headerMap.get('RIF_y') || 0],
          z: values[headerMap.get('RIF_z') || 0]
        },
        im: {
          x: values[headerMap.get('RIM_x') || 0],
          y: values[headerMap.get('RIM_y') || 0],
          z: values[headerMap.get('RIM_z') || 0]
        },
        ip: {
          x: values[headerMap.get('RIP_x') || 0],
          y: values[headerMap.get('RIP_y') || 0],
          z: values[headerMap.get('RIP_z') || 0]
        },
        elbow: {
          x: values[headerMap.get('RElbow_x') || 0],
          y: values[headerMap.get('RElbow_y') || 0],
          z: values[headerMap.get('RElbow_z') || 0]
        },
        law: {
          x: values[headerMap.get('RLAW_x') || 0],
          y: values[headerMap.get('RLAW_y') || 0],
          z: values[headerMap.get('RLAW_z') || 0]
        },
        ld: {
          x: values[headerMap.get('RLD_x') || 0],
          y: values[headerMap.get('RLD_y') || 0],
          z: values[headerMap.get('RLD_z') || 0]
        },
        lf: {
          x: values[headerMap.get('RLF_x') || 0],
          y: values[headerMap.get('RLF_y') || 0],
          z: values[headerMap.get('RLF_z') || 0]
        },
        lm: {
          x: values[headerMap.get('RLM_x') || 0],
          y: values[headerMap.get('RLM_y') || 0],
          z: values[headerMap.get('RLM_z') || 0]
        },
        lp: {
          x: values[headerMap.get('RLP_x') || 0],
          y: values[headerMap.get('RLP_y') || 0],
          z: values[headerMap.get('RLP_z') || 0]
        },
        md: {
          x: values[headerMap.get('RMD_x') || 0],
          y: values[headerMap.get('RMD_y') || 0],
          z: values[headerMap.get('RMD_z') || 0]
        },
        wrist: {
          x: values[headerMap.get('RWrist_x') || 0],
          y: values[headerMap.get('RWrist_y') || 0],
          z: values[headerMap.get('RWrist_z') || 0]
        },
        mf: {
          x: values[headerMap.get('RMF_x') || 0],
          y: values[headerMap.get('RMF_y') || 0],
          z: values[headerMap.get('RMF_z') || 0]
        },
        mm: {
          x: values[headerMap.get('RMM_x') || 0],
          y: values[headerMap.get('RMM_y') || 0],
          z: values[headerMap.get('RMM_z') || 0]
        },
        mp: {
          x: values[headerMap.get('RMP_x') || 0],
          y: values[headerMap.get('RMP_y') || 0],
          z: values[headerMap.get('RMP_z') || 0]
        },
        rd: {
          x: values[headerMap.get('RRD_x') || 0],
          y: values[headerMap.get('RRD_y') || 0],
          z: values[headerMap.get('RRD_z') || 0]
        },
        rf: {
          x: values[headerMap.get('RRF_x') || 0],
          y: values[headerMap.get('RRF_y') || 0],
          z: values[headerMap.get('RRF_z') || 0]
        },
        rm: {
          x: values[headerMap.get('RRM_x') || 0],
          y: values[headerMap.get('RRM_y') || 0],
          z: values[headerMap.get('RRM_z') || 0]
        },
        rp: {
          x: values[headerMap.get('RRP_x') || 0],
          y: values[headerMap.get('RRP_y') || 0],
          z: values[headerMap.get('RRP_z') || 0]
        },
        sh: {
          x: values[headerMap.get('RSh_x') || 0],
          y: values[headerMap.get('RSh_y') || 0],
          z: values[headerMap.get('RSh_z') || 0]
        },
        td: {
          x: values[headerMap.get('RTD_x') || 0],
          y: values[headerMap.get('RTD_y') || 0],
          z: values[headerMap.get('RTD_z') || 0]
        },
        tf: {
          x: values[headerMap.get('RTF_x') || 0],
          y: values[headerMap.get('RTF_y') || 0],
          z: values[headerMap.get('RTF_z') || 0]
        },
        tm: {
          x: values[headerMap.get('RTM_x') || 0],
          y: values[headerMap.get('RTM_y') || 0],
          z: values[headerMap.get('RTM_z') || 0]
        }
      },
      frameIndex: i - 1
    };

    frames.push(frame);
  }

  return frames;
}

// Load a CSV file
export async function loadHandMotionData(filePath: string): Promise<HandFrame[]> {
  try {
    console.log('Fetching CSV file:', filePath);
    const response = await fetch(filePath);
    console.log('fetch response status:', response.status, response.statusText);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const csvText = await response.text();
    console.log('CSV text length:', csvText.length, 'chars');
    console.log('CSV first line:', csvText.split('\n')[0]);
    
    const frames = parseHandMotionCSV(csvText);
    console.log('Parsed frame count:', frames.length);
    
    return frames;
  } catch (error) {
    console.error('Error loading hand motion data:', error);
    return [];
  }
}

// Normalize data (for convenient use in three.js)
export function normalizeHandData(frames: HandFrame[], scale: number = 0.01): HandFrame[] {
  return frames.map(frame => ({
    ...frame,
    leftHand: Object.fromEntries(
      Object.entries(frame.leftHand).map(([key, joint]) => [
        key,
        {
          x: joint.x * scale,
          y: joint.y * scale,
          z: joint.z * scale
        }
      ])
    ) as any,
    rightHand: Object.fromEntries(
      Object.entries(frame.rightHand).map(([key, joint]) => [
        key,
        {
          x: joint.x * scale,
          y: joint.y * scale,
          z: joint.z * scale
        }
      ])
    ) as any
  }));
}

// Downsample frames
export function downsampleFrames(frames: HandFrame[], originalFps: number, targetFps: number): HandFrame[] {
  if (originalFps <= targetFps) {
    console.log('No downsampling needed: source FPS is at or below the target FPS');
    return frames;
  }

  const samplingRatio = originalFps / targetFps;
  const downsampledFrames: HandFrame[] = [];

  console.log(`Downsampling: ${originalFps}fps -> ${targetFps}fps (ratio: ${samplingRatio.toFixed(2)})`);

  for (let i = 0; i < frames.length; i += samplingRatio) {
    const frameIndex = Math.floor(i);
    if (frameIndex < frames.length) {
      const frame = { ...frames[frameIndex] };
      frame.frameIndex = downsampledFrames.length; // Update to the new frame index
      downsampledFrames.push(frame);
    }
  }

  console.log(`Downsampling result: ${frames.length} -> ${downsampledFrames.length} frames`);
  return downsampledFrames;
} 