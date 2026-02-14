class DarkAmbientPad {
  constructor() {
    // Keep the shared effects
    this.vibrato = new Tone.Vibrato({
      frequency: 0.5,
      depth: 0.3,
      wet: 1.0
    });
    
    this.chorus = new Tone.Chorus({
      frequency: 0.2,
      delayTime: 4,
      depth: 0.5,
      spread: 180,
      wet: 0.4
    }).start();
    
    this.filter = new Tone.Filter({
      frequency: 600,
      type: "lowpass",
      rolloff: -12
    });
    
    this.filterLFO = new Tone.LFO({
      frequency: 0.05,
      min: 400,
      max: 900,
      type: "sine"
    });
    this.filterLFO.connect(this.filter.frequency);
    this.filterLFO.start();
    
    // Create a MIXER for all voices
    this.mixer = new Tone.Gain(0.5 * MASTER_VOLUME);
    
    // Chain shared effects: vibrato -> filter -> chorus -> mixer
    this.vibrato.chain(this.filter, this.chorus, this.mixer, Tone.getDestination());
    
    // Store individual synths per note
    this.voices = new Map();
  }
  
  createVoice(note, volumeGate) {
    // Create a dedicated synth for this note
    const synth = new Tone.FMSynth({
      harmonicity: 1.01,
      modulationIndex: 2,
      oscillator: { type: "sine" },
      envelope: {
        attack: 2.5,
        decay: 1.0,
        sustain: 0.8,
        release: 5.0
      },
      modulation: { type: "triangle" },
      modulationEnvelope: {
        attack: 4.0,
        decay: 0.1,
        sustain: 1.0,
        release: 4.0
      }
    });
    
    // Chain: synth -> volumeGate -> shared effects
    synth.connect(volumeGate);
    volumeGate.connect(this.vibrato);
    
    // Trigger the note and leave it on
    synth.triggerAttack(note);
    
    this.voices.set(note, synth);
  }
  
  dispose() {
    this.voices.forEach(synth => synth.dispose());
    this.vibrato.dispose();
    this.filter.dispose();
    this.filterLFO.dispose();
    this.chorus.dispose();
    this.mixer.dispose();
  }
}
