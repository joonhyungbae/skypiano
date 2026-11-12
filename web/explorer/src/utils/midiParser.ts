import { Midi } from '@tonejs/midi';

// MIDI note data interface
export interface MidiNote {
  note: number; // MIDI note number (0-127)
  startTime: number; // Start time (seconds)
  duration: number; // Duration (seconds)
  velocity: number; // Velocity (0-127)
  channel: number; // MIDI channel
}

// MIDI data interface
export interface MidiData {
  notes: MidiNote[];
  totalDuration: number; // Total length (seconds)
  ticksPerQuarter: number; // Ticks per quarter note
  tempo: number; // BPM
}

// Parse MIDI data
export async function parseMidiFile(arrayBuffer: ArrayBuffer): Promise<MidiData> {
  try {
    console.log('Parsing MIDI file');
    
    // Convert ArrayBuffer to Uint8Array
    const uint8Array = new Uint8Array(arrayBuffer);
    
    // Parse the MIDI file with @tonejs/midi
    const midi = new Midi(uint8Array);
    
    console.log('MIDI parsed:', {
      name: midi.name,
      tracks: midi.tracks.length,
      duration: midi.duration,
      ticksPerQuarter: midi.header.ppq
    });

    const notes: MidiNote[] = [];
    
    // Collect notes from all tracks
    midi.tracks.forEach((track, trackIndex) => {
      console.log(`Track ${trackIndex}:`, {
        name: track.name,
        instrument: track.instrument?.name,
        notes: track.notes.length
      });
      
      track.notes.forEach((note) => {
        notes.push({
          note: note.midi,
          startTime: note.time,
          duration: note.duration,
          velocity: Math.round(note.velocity * 127), // Scale 0-1 to 0-127
          channel: track.channel || 0
        });
      });
    });

    // Sort by start time
    notes.sort((a, b) => a.startTime - b.startTime);

    const result: MidiData = {
      notes,
      totalDuration: midi.duration,
      ticksPerQuarter: midi.header.ppq || 480,
      tempo: midi.header.tempos.length > 0 ? midi.header.tempos[0].bpm : 120
    };

    console.log('Parse result:', {
      totalNotes: notes.length,
      duration: result.totalDuration,
      tempo: result.tempo
    });

    return result;
  } catch (error) {
    console.error('MIDI parsing error:', error);
    throw new Error(`Could not parse MIDI file: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

// Convert a MIDI note number to a note name
export function midiNoteToName(midiNote: number): string {
  const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const octave = Math.floor(midiNote / 12) - 1;
  const noteName = noteNames[midiNote % 12];
  return `${noteName}${octave}`;
}

// Format seconds as minutes:seconds
export function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}