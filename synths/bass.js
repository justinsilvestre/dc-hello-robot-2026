
class BassSynth {
  constructor() {
    // Main bass voice - sawtooth for brightness
    this.mainSynth = new Tone.MonoSynth({
      oscillator: {
        type: "sine",
      },
      envelope: {
        attack: 0.01,
        decay: 0.2,
        sustain: 0.6,
        release: 0.4,
      },
      filterEnvelope: {
        attack: 0.005,
        decay: 0.1,
        sustain: 0.4,
        release: 0.3,
        baseFrequency: 200,
        octaves: 1,
      },
    });

    // Sub-bass voice - sine wave one octave down
    this.subSynth = new Tone.MonoSynth({
      oscillator: {
        type: "sine",
      },
      envelope: {
        attack: 0.01,
        decay: 0.3,
        sustain: 0.8,
        release: 0.5,
      },
      filter: {
        type: "lowpass",
        frequency: 50,
        Q: 1,
      },
    });

    // Saturation for warmth and harmonics
    this.distortion = new Tone.Distortion({
      distortion: 0.4,
      wet: 0.3,
    });

    // EQ to boost sub frequencies
    this.eq = new Tone.EQ3({
      low: 6,
      mid: 0,
      high: -3,
      lowFrequency: 100,
      highFrequency: 2500,
    });

    // Compression to glue it together
    this.compressor = new Tone.Compressor({
      threshold: -20,
      ratio: 4,
      attack: 0.003,
      release: 0.1,
    });

    // Subtle chorus for width
    this.chorus = new Tone.Chorus({
      frequency: 2,
      delayTime: 2.5,
      depth: 0.3,
      wet: 0.2,
    }).start();

    // Short reverb
    this.reverb = new Tone.Reverb({
      decay: 0.8,
      wet: 0.1,
    });

    // Master gain
    this.gain = new Tone.Gain(0.025 * MASTER_VOLUME);

    // Connect main synth chain
    this.mainSynth.chain(
      this.distortion,
      this.chorus,
      this.eq,
      this.compressor,
      this.reverb,
      this.gain,
      Tone.getDestination()
    );

    // Connect sub synth directly (bypass effects for clarity)
    this.subSynth.chain(this.compressor, this.gain);

    this.reverb.generate();
  }

  async start() {
    await Tone.start();
  }

  playNote(
    note,
    duration,
    time
  ) {
    note = Tone.Frequency(note).transpose(-12).toNote();
    // Play main note
    this.mainSynth.triggerAttackRelease(note, duration, time);
    
    // Play sub-bass one octave lower
    const subNote = Tone.Frequency(note).transpose(-24).toNote();
    this.subSynth.triggerAttackRelease(subNote, duration, time);
    if (LOG_PLAYBACK) {
      console.log(`Playing bass note: ${note} at time ${time} for duration ${duration}`);
      console.log(`Playing subbass note: ${subNote} at time ${time} for duration ${duration}`);
    }
  }

  triggerAttack(note, time) {
    if (LOG_PLAYBACK) {
      console.log(`Playing attack for bass note: ${note} at time ${time}`);
      console.log(`Playing attack for subbass note: ${Tone.Frequency(note).transpose(-24).toNote()} at time ${time}`);
    }
    this.mainSynth.triggerAttack(note, time);
    const subNote = Tone.Frequency(note).transpose(-36).toNote();
    this.subSynth.triggerAttack(subNote, time);
  }

  triggerRelease(time) {
    console.log(`Playing release for bass synths at time ${time}`);
    console.log(`Playing release for subbass synth at time ${time}`);
    this.mainSynth.triggerRelease(time);
    this.subSynth.triggerRelease(time);
  }

  dispose() {
    this.mainSynth.dispose();
    this.subSynth.dispose();
    this.distortion.dispose();
    this.eq.dispose();
    this.compressor.dispose();
    this.chorus.dispose();
    this.reverb.dispose();
    this.gain.dispose();
  }
}
