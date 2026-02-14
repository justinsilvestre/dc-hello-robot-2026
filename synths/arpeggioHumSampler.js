class ArpeggioHumSampler {
  constructor(sampleUrl) {
    this.loaded = false;
    this.bufferDuration = 0;
    
    // Create MULTIPLE players for polyphony
    this.numVoices = 4;
    this.players = [];
    this.nextVoice = 0;
    
    let loadCount = 0;
    
    for (let i = 0; i < this.numVoices; i++) {
      let player = new Tone.Player({
        url: sampleUrl,
        loop: false,
        autostart: false,
        fadeIn: 0.05,
        fadeOut: 0.5,
        onload: () => {
          loadCount++;
          if (loadCount === this.numVoices) {
            this.loaded = true;
            this.bufferDuration = player.buffer.duration;
            console.log("Arpeggio loaded");
            this.start();
          }
        }
      });
      this.players.push(player);
    }
    
    this.gain = new Tone.Gain(0.0075 * MASTER_VOLUME);
    this.filter = new Tone.Filter({
      type: "lowpass",
      frequency: 6000
    });
    
    this.compressor = new Tone.Compressor({
      threshold: -18,
      ratio: 4,
      attack: 0.01,
      release: 0.25
    }).toDestination();
    
    // Connect all players to the same effects chain
    this.players.forEach(player => {
      player.chain(
        this.gain,
        this.filter,
        this.compressor
      );
    });
    
    this.repeatId = null;
    this.probability = 0.4;
  }
  
  start() {
    if (!this.loaded || this.repeatId) return;
    
    this.repeatId = Tone.Transport.scheduleRepeat((time) => {
      if (!this.loaded) return;
      
      if (Math.random() < this.probability) {
        // Use round-robin voice allocation
        let player = this.players[this.nextVoice];
        this.nextVoice = (this.nextVoice + 1) % this.numVoices;
        
        // Stop the player first if it's already playing
        if (player.state === "started") {
          player.stop(time);
        }
        
        const offset = random(0, Math.max(this.bufferDuration - 6, 0));
        const duration = 6;
        
        player.start(time, offset, duration);
        player.stop(time + duration);  // Removed the +0.5, fadeOut handles it
      }
    }, "16n");
  }
  
  update(entropy) {
    let targetFilterFreq = map(entropy, 0, 1, 1000, 6000);
    this.filter.frequency.rampTo(targetFilterFreq, 0.1);
  }
  
  stop() {
    if (this.repeatId) {
      Tone.Transport.clear(this.repeatId);
      this.repeatId = null;
    }
    this.players.forEach(p => {
      if (p.state === "started") p.stop();
    });
  }
  
  dispose() {
    this.stop();
    this.players.forEach(p => p.dispose());
    this.filter.dispose();
    this.gain.dispose();
    this.compressor.dispose();
  }
}
