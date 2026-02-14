class AbstractHummingSampler {
  constructor(sampleUrl) {
    this.loaded = false;
    this.bufferDuration = 0;
    
    // Create 8 players for thick layering
    this.players = [];
    let loadCount = 0;
    const numPlayers = 8;
    
    for (let i = 0; i < numPlayers; i++) {
      let p = new Tone.Player({
        url: sampleUrl,
        loop: true,  // CONTINUOUS LOOP
        autostart: false,
        onload: () => {
          loadCount++;
          if (loadCount === numPlayers) {
            this.loaded = true;
            this.bufferDuration = p.buffer.duration;
            console.log('Humming loaded, duration:', this.bufferDuration);
          }
        }
      });
      this.players.push(p);
    }
    
    // Create individual effect chains for each player
    this.playerChains = [];
    
    this.players.forEach((player, i) => {
      // Individual pitch shift for each layer (harmonic cloud)
      let pitchShift = new Tone.PitchShift({
        pitch: [-19, -12, -7, -5, 0, 5, 7, 12][i], // Spread across octaves
        windowSize: 0.4,
        wet: 1.0
      });
      
      // Individual playback rate for detuning/chorus effect
      // player.playbackRate = 1.0 - (0.05 * i);
      
      // Slowly evolving panning for spatial movement
      let panner = new Tone.Panner(Math.sin(i) * 0.7);
      
      // Individual volume control for waves
      let gain = new Tone.Gain(0); // Start silent
      
      this.playerChains.push({
        // pitchShift,
        panner,
        gain
      });
      
      player.chain(pitchShift, panner, gain);
    });
    
    // SHARED MASTER EFFECTS for cohesion
    
    // Multi-tap delay for space siren quality
    this.delay1 = new Tone.FeedbackDelay({
      delayTime: "4n.",
      feedback: 0.5,
      wet: 0.3
    });
    
    this.delay2 = new Tone.FeedbackDelay({
      delayTime: "8t",
      feedback: 0.4,
      wet: 0.2
    });
    
    // Shimmer reverb for heavenly quality
    this.shimmerReverb = new Tone.Reverb({
      decay: 6,
      wet: 0.7
    });
    
    // Chorus for thickness
    this.chorus = new Tone.Chorus({
      frequency: 0.3,
      delayTime: 8,
      depth: 0.6,
      spread: 180,
      wet: 0.5
    }).start();

    this.filter = new Tone.Filter({
      type: "lowpass",
      frequency: 6000
    });
    
    // Frequency shifter for otherworldly quality
    this.freqShift = new Tone.FrequencyShifter({
      frequency: 0,  // Will modulate this
      wet: 0.4
    });
    
    // Tremolo for pulsing waves
    this.tremolo = new Tone.Tremolo({
      frequency: 0.15,  // Slow pulse
      depth: 0.4,
      spread: 180
    }).start();
    
    // Stereo widener for immersive space
    this.stereoWidener = new Tone.StereoWidener(0.8);
    
    // Compressor to glue everything
    this.compressor = new Tone.Compressor({
      threshold: -20,
      ratio: 3,
      attack: 0.1,
      release: 0.3
    });
    
    this.mixer = new Tone.Gain(0.5);
    
    // Connect all player gains to master chain
    this.playerChains.forEach(chain => {
      chain.gain.chain(
        this.delay1,
        this.delay2,
        this.filter,
        this.chorus,
        this.shimmerReverb,
        this.tremolo,
        this.stereoWidener,
        this.compressor,
        this.mixer,
        Tone.getDestination()
      );
    });
    
    // Wave automation state
    this.wavePhase = 0;
    this.activeWaves = 0;
  }
  
  // Call this continuously in draw() for eternal waves
  updateWaves(entropy, frameCount) {
    if (!this.loaded) return;
    
    this.wavePhase += 0.001;
    
    // Modulate tremolo speed based on entropy
    this.tremolo.frequency.rampTo(
      map(entropy, 0, 1, 0.08, 13),
      0.1
    );
    
    // Evolving panning for spatial movement
    this.playerChains.forEach((chain, i) => {
      let panPhase = this.wavePhase + i * 0.3;
      chain.panner.pan.rampTo(
        Math.sin(panPhase) * 0.8,
        3.0
      );
    });
    
    // WAVE TRIGGERING - layers fade in/out creating eternal waves
    this.players.forEach((player, i) => {
      let chain = this.playerChains[i];
      
      // Each layer has its own phase offset for staggered waves
      let layerPhase = this.wavePhase + i * 0.7;
      let wave = (Math.sin(layerPhase) + 1) / 2; // 0 to 1
      
      // Map to volume (some layers always present, others pulse)
      let targetVol = map(
        wave,
        0, 1,
        i < 3 ? 0.01 : 0,  // First 3 layers always slightly on
        0.05
      );

      // Add entropy influence
      targetVol *= map(entropy, 0, 1, 0.5, 2.0);
      targetVol *= MASTER_VOLUME;
      // Smooth volume changes for wave-like swells
      chain.gain.gain.rampTo(targetVol, 8.0); // 8 second transitions
      
      let targetFilterFreq = map(entropy, 0, 1, 1000, 6000);
      this.filter.frequency.rampTo(targetFilterFreq, 0.1);

      // Start/stop players based on wave state
      if (targetVol > 0.01 && player.state !== "started") {
        // Random offset in sample for variation
        let offset = random(0, this.bufferDuration * 0.3);
        player.start(Tone.now(), offset);
      } else if (targetVol < 0.01 && player.state === "started") {
        player.stop(Tone.now() + 4); // Fade out over 4 seconds
      }
    });
  }
  
  // Trigger a sudden wave burst (for expansion/contraction events)
  triggerWaveBurst(intensity = 1.0) {
    if (!this.loaded) return;
    
    this.players.forEach((player, i) => {
      let chain = this.playerChains[i];
      
      // Surge all layers
      let burstVol = random(0.1, 0.25) * intensity;
      chain.gain.gain.rampTo(burstVol, 0.5);
      
      // Then decay
      chain.gain.gain.rampTo(0.05, 6.0, Tone.now() + 2);
      
      if (player.state !== "started") {
        player.start(Tone.now(), random(0, this.bufferDuration * 0.5));
      }
    });
    
    // Dramatic frequency shift
    // this.freqShift.frequency.rampTo(-20, 1.0);
    // this.freqShift.frequency.rampTo(5, 8.0, Tone.now() + 1);
  }
  
  stopAll() {
    this.players.forEach(p => {
      if (p.state === "started") {
        p.stop(Tone.now() + 4); // Graceful fade
      }
    });
  }
  
  dispose() {
    this.stopAll();
    this.players.forEach(p => p.dispose());
    this.playerChains.forEach(chain => {
      chain.pitchShift.dispose();
      chain.panner.dispose();
      chain.gain.dispose();
    });
    this.delay1.dispose();
    this.delay2.dispose();
    this.freqShift.dispose();
    this.shimmerReverb.dispose();
    this.chorus.dispose();
    this.tremolo.dispose();
    this.stereoWidener.dispose();
    this.compressor.dispose();
    this.mixer.dispose();
  }
}
