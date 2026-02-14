
class ProximityPad {
  constructor() {
    // Gentle wobble for organic unease
    this.vibrato = new Tone.Vibrato({
      frequency: 0.4,
      depth: 0.2,
      wet: 0.8
    });
    
    // Subtle detuning for dreamlike quality
    this.chorus = new Tone.Chorus({
      frequency: 0.3,
      delayTime: 6,
      depth: 0.6,
      spread: 180,
      wet: 0.5
    }).start();
    
    // Warm filter that breathes
    this.filter = new Tone.Filter({
      frequency: 1200,
      type: "lowpass",
      rolloff: -12,
      Q: 1.5
    });
    
    this.filterLFO = new Tone.LFO({
      frequency: 0.06,
      min: 600,
      max: 1400,
      type: "sine"
    });
    this.filterLFO.connect(this.filter.frequency);
    this.filterLFO.start();
    
    // Lush reverb for nostalgia
    this.reverb = new Tone.Reverb({
      decay: 10,
      wet: 0.55
    });
    
    // Subtle tremolo for fragility
    this.tremolo = new Tone.Tremolo({
      frequency: 0.5,
      depth: 0.25,
      spread: 120
    }).start();
    
    // Phaser for slow movement/unease
    this.phaser = new Tone.Phaser({
      frequency: 0.15,
      octaves: 3,
      baseFrequency: 350,
      Q: 2,
      wet: 0.3
    });
    
    // Stereo width for immersion
    this.stereoWidener = new Tone.StereoWidener(0.5);
    
    // Compression for glue and presence
    this.compressor = new Tone.Compressor({
      threshold: -20,
      ratio: 3,
      attack: 0.1,
      release: 0.3
    });
    
    this.mixer = new Tone.Gain(0.65);
    
    // Chain for warmth and depth
    this.vibrato.chain(
      this.chorus,
      this.phaser,
      this.filter,
      this.tremolo,
      // this.reverb,
      this.stereoWidener,
      this.compressor,
      this.mixer,
      Tone.getDestination()
    );
    
    this.voices = new Map();
  }
  
  createVoice(note, volumeGate) {
    const synth = new Tone.FMSynth({
      harmonicity: 1.5,     // Musical perfect fifth relationship
      modulationIndex: 2,   // Rich but controlled harmonics
      oscillator: { 
        type: "sine"        // Pure, warm tone
      },
      envelope: {
        attack: 3.5,
        decay: 1.2,
        sustain: 0.85,
        release: 5.0
      },
      modulation: { 
        type: "triangle"    // Softer modulation
      },
      modulationEnvelope: {
        attack: 4.5,
        decay: 0.8,
        sustain: 0.95,
        release: 4.5
      }
    });
    
    synth.connect(volumeGate);
    volumeGate.connect(this.vibrato);
    if (LOG_PLAYBACK)
    console.log(`Playing ProximityPad note: ${note}`);
    synth.triggerAttack(Tone.Frequency(note).transpose(12));
    
    this.voices.set(note, synth);
  }
  
  dispose() {
    this.voices.forEach(synth => synth.dispose());
    this.vibrato.dispose();
    this.chorus.dispose();
    this.filter.dispose();
    this.filterLFO.dispose();
    this.reverb.dispose();
    this.tremolo.dispose();
    this.phaser.dispose();
    this.stereoWidener.dispose();
    this.compressor.dispose();
    this.mixer.dispose();
  }
}
