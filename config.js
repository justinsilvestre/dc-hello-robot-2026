// Music 
const SCALE_ROOT_NOTE = "C#";
const SCALE_MODE = "minor";
const SCALE_OCTAVE = 3;

const SCALE_NOTES        = Tonal.Scale.get(`${SCALE_ROOT_NOTE}${SCALE_OCTAVE} ${SCALE_MODE}`).notes;
const SCALE_OCTAVE_ABOVE = Tonal.Scale.get(`${SCALE_ROOT_NOTE}${SCALE_OCTAVE + 1} ${SCALE_MODE}`).notes;

const MASTER_VOLUME = 7.5;

const EXPANSION_CHORDS = [
  [SCALE_NOTES[5], SCALE_NOTES[0], SCALE_NOTES[2]],
  [SCALE_NOTES[2], SCALE_NOTES[5], SCALE_NOTES[0]],
  [SCALE_NOTES[4], SCALE_NOTES[6], SCALE_NOTES[1]],
  [SCALE_NOTES[3], SCALE_NOTES[5], SCALE_NOTES[0], SCALE_NOTES[2]],
];
const RESONANCE_SLICE_CHORD = [SCALE_NOTES[8], SCALE_NOTES[4], SCALE_NOTES[6], SCALE_NOTES[0], SCALE_NOTES[2]];
const CONTRACTION_CHORD     = [SCALE_NOTES[0], SCALE_NOTES[2], SCALE_NOTES[6]];

// Attractor
const ITERATIONS_PER_FRAME = 10000;
const G_STRENGTH           = 0.1;
const SOFTENING            = 0.0002;
const DENSITY_THRESHOLD = ITERATIONS_PER_FRAME * 0.05;

const MAX_VISITORS = 4;

// Visitor visual configuration
const VISITOR_RADIUS = 40;           // Increased from 40
const VISITOR_OUTLINE_THICKNESS = 3; // Configurable outline thickness
const VISITOR_OUTLINE_OPACITY = 30;  // Outline opacity

// Particle visual configuration
const PARTICLE_SIZE = 2;           // Particle stroke weight (was 0.9)

const MANUAL_VISITOR_ID = 'manual-test-visitor';

const LOG_PLAYBACK = true; // log when synths are triggered for debugging