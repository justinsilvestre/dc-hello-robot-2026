class PoemSampler {
  constructor() {
    this.isLoaded = false;
    
    // Use 'this.sampler' instead of bare 'sampler'
    this.sampler = new Tone.Sampler({
      urls: {
        A1: "http://localhost:8080/p1",  // Use full URL with correct port and endpoint
        B1: "http://localhost:8080/p2",
        C1: "http://localhost:8080/p3",
        D1: "http://localhost:8080/p4",
      },
      onload: () => {
        this.isLoaded = true;
        console.log("Poem samples loaded.");
      }
    });


    this.notes = ['A1', 'B1', 'C1', 'D1'];
    this.noteIndex = 0;
    
    this.reverb = new Tone.Reverb({
      decay: 2,
      wet: 0.3
    });

    this.delay = new Tone.FeedbackDelay({
      feedback: 0.3,
      delayDuration: "8n.",
      wet: 0.4
    });

    this.filter = new Tone.Filter({
      frequency: 6000,
      type: "lowpass"
    });

    this.filterLFO = new Tone.LFO({
      frequency: 0.1,
      min: 2000,
      max: 6000
    });
    this.filterLFO.connect(this.filter.frequency);
    this.filterLFO.start();

    this.panner = new Tone.Panner(0.0);
    // Stereo widener for immersive space
    this.stereoWidener = new Tone.StereoWidener(0.8);
    
    // Compressor to glue everything
    this.compressor = new Tone.Compressor({
      threshold: -20,
      ratio: 3,
      attack: 0.1,
      release: 0.3
    });
    
    this.gain = new Tone.Gain(0.45 * MASTER_VOLUME);

    this.sampler.chain(
      this.filter,
      this.reverb,
      this.stereoWidener,
      // this.delay,
      this.compressor,
      this.panner,
      this.gain,
      Tone.getDestination()
    );

    this.lastTriggered = 0;
    this.cooldown = 200; // 2 second cooldown between triggers
  }
  
    
  update(screenCentroidX) {
    this.panner.pan.rampTo(constrain(map(screenCentroidX, width * 0.45, width * 0.55, -1, 1), -1, 1), 0.1);
  }

  // Trigger the next poem line
  trigger() {
    if (!this.isLoaded) return;
    
    if (Date.now() - this.lastTriggered < this.cooldown) return;

    if (LOG_PLAYBACK)
    console.log(`Playing poem sampler note: ${this.notes[this.noteIndex]} at time ${Tone.now()}`);
    this.sampler.triggerAttackRelease(this.notes[this.noteIndex], "1n");

    this.noteIndex += 1;
    this.noteIndex %= 4;
    this.lastTriggered = Date.now();
  }
  
  // Trigger specific line
  triggerLine(lineNumber) {
    if (!this.isLoaded) return;
    if (lineNumber < 1 || lineNumber > 4) return;
    
    const notes = ['A1', 'B1', 'C1', 'D1'];
    this.sampler.triggerAttackRelease(notes[lineNumber - 1], "1n");
  }
  
  dispose() {
    this.sampler.dispose();
  }
}
