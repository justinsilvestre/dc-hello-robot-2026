class ProximityMelody {
  constructor() {
    this.synth = new Tone.PolySynth(Tone.FMSynth, {
      maxPolyphony: 4,
      harmonicity: 1,
      modulationIndex: 1.2,
      oscillator: { type: "sine" },
      envelope: {
        attack: 0.02,
        decay: 0.8,
        sustain: 0.1,
        release: 2.5
      },
      modulation: { type: "sine" },
      modulationEnvelope: {
        attack: 0.01,
        decay: 0.4,
        sustain: 0.2,
        release: 1.5
      }
    });
    
    this.reverb = new Tone.Reverb({
      decay: 4,
      wet: 0.4
    });
    
    this.chorus = new Tone.Chorus({
      frequency: 0.5,
      delayTime: 3,
      depth: 0.3,
      wet: 0.25
    }).start();

    this.pingPong = new Tone.PingPongDelay({
      feedback: 0.4,
      delayTime: "16t"
    });
    
    this.filter = new Tone.Filter({
      type: "lowpass",
      frequency: 2500,
      rolloff: -12
    });
    
    this.gain = new Tone.Gain(0.0);
    
    this.synth.chain(
      this.chorus,
      this.filter,
      this.pingPong,
      this.reverb,
      this.gain,
      Tone.getDestination()
    );
    
    this.melodyPatterns = [
      [SCALE_OCTAVE_ABOVE [4], SCALE_OCTAVE_ABOVE [2], SCALE_OCTAVE_ABOVE [0]],
      [SCALE_OCTAVE_ABOVE [0], SCALE_OCTAVE_ABOVE [2], SCALE_OCTAVE_ABOVE [5]],
      [SCALE_OCTAVE_ABOVE [5], SCALE_OCTAVE_ABOVE [4], SCALE_OCTAVE_ABOVE [3], SCALE_OCTAVE_ABOVE [2]], 
      [SCALE_OCTAVE_ABOVE [0], SCALE_OCTAVE_ABOVE [4]],
      [SCALE_OCTAVE_ABOVE [2], SCALE_OCTAVE_ABOVE [6], SCALE_OCTAVE_ABOVE [4]],
    ];
    
    this.currentPattern = random(this.melodyPatterns);
    this.repeatId = null;
    this.proximityLevel = 1;
  }
  
  start() {
    if (this.repeatId) return;
    
    let noteIndex = 0;
    
    this.repeatId = Tone.Transport.scheduleRepeat((time) => {
      const activePositions = [0, 5, 10, 14].slice(0, this.currentPattern.length);
      const position = noteIndex % 16;
      
      if (activePositions.includes(position)) {
        const patternIndex = activePositions.indexOf(position);
        let note = this.currentPattern[patternIndex];
        
        if (this.proximityLevel > 0.1) {
          const velocity = map(this.proximityLevel, 0.1, 1, 0.2, 0.6);

          if (random(1) > 0.7) note = Tone.Frequency(note).transpose(random() > 0.5 ? 12 : -12);
          
          if (LOG_PLAYBACK)
          console.log(`Playing ProximityMelody note: ${note} at time ${time}`);
          this.synth.triggerAttackRelease(note, "8n", time);
        }
      }
      
      noteIndex++;
      
      if (noteIndex % 32 === 0 && Math.random() < 0.3) {
        this.currentPattern = random(this.melodyPatterns);
      }
      
    }, "16n");
  }

  update(visitorIndex1, visitorIndex2, distance) {
    const volume = map(constrain(distance, 0, 200), 0, 200, 0.2, 0.0) * MASTER_VOLUME;
    this.gain.gain.rampTo(volume, 0.1);
  }
  
  stop() {
    if (this.repeatId) {
      Tone.Transport.clear(this.repeatId);
      this.repeatId = null;
    }
  }
  
  dispose() {
    this.stop();
    this.synth.dispose();
    this.reverb.dispose();
    this.chorus.dispose();
    this.filter.dispose();
    this.gain.dispose();
  }
}

function random(array) {
  return array[Math.floor(Math.random() * array.length)];
}