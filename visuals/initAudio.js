async function initAudio(state) {
  try {
    console.log("Starting Audio Context...");
    Tone.setContext(new Tone.Context({
      latencyHint: "playback",
      lookAhead: 0.1,
      latencyHint: 0.5,
      updateInterval: 0.05,
      sampleRate: 44100
    }));
    console.log("Audio Context Configured:", Tone.getContext());
    await Tone.start();
      
        

    console.log("🎵 Starting audio initialization...");
    
    const initPromise = (async () => {
      // Check if Tone is properly started
      if (Tone.context.state !== 'running') {
        console.warn("⚠️ Tone context not running, attempting to resume...");
        await Tone.context.resume();
        
        // Wait a bit for context to stabilize
        await new Promise(resolve => setTimeout(resolve, 100));
        
        if (Tone.context.state !== 'running') {
          throw new Error(`Tone context state is ${Tone.context.state}, expected 'running'`);
        }
      }
      
      console.log("Setting Transport BPM...");
      Tone.Transport.bpm.value = 120;

      console.log("Initializing synths...");
      // Test synth creation first
      const testSynth = new Tone.Oscillator(440, "sine");
      testSynth.dispose();
      console.log("✅ Synth creation test passed");
      
      const darkPad = new DarkAmbientPad(); 
      SCALE_NOTES.forEach(note => {
        darkPad.createVoice(note, new Tone.Gain(0));
        state.resonanceSlices.push(new ResonanceSlice(note));
      });

      console.log("Loading samples...");
      hummingWavesSampler    = new AbstractHummingSampler("http://localhost:8080/sample");  // Use full URL
      hummingArpeggioSampler = new ArpeggioHumSampler("http://localhost:8080/sample");     // Use full URL
      poemSampler            = new PoemSampler();

      console.log("Setting up proximity systems...");
      proximityPad = new ProximityPad();
      for (let i = 0; i < state.visitors.length; i += 1) {
        state.visitors[i].volumeGate = new Tone.Gain(0.2);
        proximityPad.createVoice(SCALE_NOTES[i], state.visitors[i].volumeGate);
      }

      proximityMelody = new ProximityMelody();
      proximityMelody.start();

      console.log("Initializing bass synth...");
      bass = new BassSynth();
      
      console.log("Starting Transport...");
      Tone.Transport.start();
      
      // Verify transport is running
      await new Promise(resolve => setTimeout(resolve, 100));
      console.log("Transport state:", Tone.Transport.state);

      return {
        darkPad,
        hummingWavesSampler,
        hummingArpeggioSampler,
        poemSampler,
        proximityPad,
        proximityMelody,
        bass
      }
    })();
  

    audioInitialized = true;

    return await initPromise;
    
  } catch (error) {
    console.error("❌ Audio initialization failed:", error);
    
    // Add more diagnostic info
    console.log("🔍 Audio Context Details:");
    console.log("- State:", Tone.context?.state);
    console.log("- Sample Rate:", Tone.context?.sampleRate);
    console.log("- Current Time:", Tone.context?.currentTime);
    console.log("- Destination:", Tone.context?.destination);
    
    audioInitialized = false;
    throw error; // Re-throw so the caller knows it failed
  }
}